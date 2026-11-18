"""Score native eight-transition forecasts on freshly collected policy trajectories."""

import numpy as np
import torch

from evaluate_saved_wm_forecasts import GROUPS, WIDTHS, forecast, numpy_groups
from push_recovery_contract import sha256, write_json


def score_forecasts(output, records, world_model, context, runner_cfg):
    if not torch.equal(context, context[:1].expand_as(context)):
        raise ValueError("The native probe needs one shared fixed simulator context")
    rep = world_model.representation
    nominals = np.array([getattr(runner_cfg.wm, rep.LOSS_WEIGHT_CONFIG[k]) for k in GROUPS])
    weights = WIDTHS / np.square(nominals) / WIDTHS.sum()
    total = sum(max(0, r["steps"] - 8) for r in records)
    if total <= 0:
        raise ValueError("No complete eight-transition window in the native rollouts")
    path = output / "state-forecast-errors.npy"
    bank = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(total, 8, 4))
    cursor = 0
    order = []
    independent_checks = 0
    with torch.inference_mode():
        for record in records:
            count = max(0, record["steps"] - 8)
            order.append({"index": record["index"], "steps": record["steps"],
                          "offset": cursor, "windows": count})
            if not count:
                continue
            with np.load(output / record["trajectory"], allow_pickle=False) as case:
                states = np.moveaxis(np.lib.stride_tricks.sliding_window_view(
                    case["full_state"], 9, axis=0), -1, 1)
                actions = np.moveaxis(np.lib.stride_tricks.sliding_window_view(
                    case["actions"], 8, axis=0), -1, 1)[:count]
                if len(states) != count or len(actions) != count:
                    raise RuntimeError("A forecast window is missing a state or action")
                for start in range(0, count, 4096):
                    stop = min(start + 4096, count)
                    real = torch.as_tensor(states[start:stop].copy(), device=context.device)
                    commands = torch.as_tensor(actions[start:stop].copy(), device=context.device)
                    predicted, aux, errors = forecast(world_model.dynamics, rep, real, commands,
                                                       context[:1].expand(stop-start, -1))
                    bank[cursor + start:cursor + stop] = errors.cpu().numpy()
                    if start == 0:
                        predicted_full = rep.reconstruct_full_state(aux[0], predicted[0])
                        check = numpy_groups(predicted_full.cpu().numpy(), real[:1, 1:].cpu().numpy())
                        if not np.allclose(check, errors[:1].cpu().numpy(), rtol=1e-4, atol=2e-6):
                            raise RuntimeError("Native and independent state error disagree")
                        independent_checks += 1
            cursor += count
    if cursor != total:
        raise RuntimeError("Forecast output missing declared windows")
    bank.flush()
    del bank
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid errors in the saved forecast array")
    group_means = values.mean(axis=0, dtype=np.float64)
    scores = group_means @ weights
    result = {
        "status": "fresh_native_rollout_state_forecasts_verified",
        "horizon_transitions": 8, "windows": total, "window_order": order,
        "independent_first_window_checks": independent_checks,
        "state_group_order": GROUPS, "state_group_dimensions": WIDTHS.tolist(),
        "nominal_errors": nominals.tolist(), "group_mse_by_horizon": group_means.tolist(),
        "normalized_state_mse_by_horizon": scores.tolist(),
        "eighth_transition_normalized_state_mse": float(scores[-1]),
        "mean_normalized_state_mse_over_horizons": float(scores.mean()),
        "simulation_actions": "same frozen actor in actual physics at this checkpoint",
        "forecast_timing": "postprocessing of newly collected deterministic on-policy trajectories",
        "terminal_sample_convention": "nine consecutive pre-action states; terminal post-state excluded",
        "array_sha256": sha256(path),
        "author_exact_prediction_error_aggregation": False,
    }
    write_json(output / "state-forecast.json", result)
    return result
