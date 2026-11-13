"""Exercise the released PPO-WM/RSL termination boundary on CPU.

Uses the actual native soft-termination function and PPO rollout/update code,
with synthetic observations and rewards. This is an interface diagnostic,
not a G1 rollout, learned-model experiment or paper mechanism comparison.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path


def file_record(module):
    path = Path(inspect.getfile(module))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_probe():
    import torch
    import rsl_rl.algorithms.ppo as rsl_ppo
    import rsl_rl.storage.rollout_storage as rsl_storage
    import supermimic.envs.world_model_env as wm_module
    import supermimic.runners.ppo_runner as native
    from supermimic.envs.motion_tracking_env import MotionTrackingEnv
    from tensordict import TensorDict

    torch.set_num_threads(2)
    # Physical quantities below only exercise the actual termination function;
    # no simulator is created and the observation dimensions are synthetic.
    height_errors = torch.tensor([0., .125, .2, .249, .25, .275, .3, .5])
    n = len(height_errors)
    wm = MotionTrackingEnv.__new__(MotionTrackingEnv)
    wm._anchor_index = 0
    wm._anchor_pos_threshold = .25
    wm._anchor_ori_threshold = .8
    wm._termination_softness = .05
    aux = torch.zeros(n, 7)
    aux[:, 3] = 1.
    ref_pos = torch.zeros(n, 1, 3)
    ref_pos[:, 0, 2] = height_errors
    ref_quat = torch.zeros(n, 1, 4)
    ref_quat[:, :, 0] = 1.
    probabilities = wm._compute_term_prob(aux, ref_pos, ref_quat)

    class SyntheticWorldModelEnv:
        def obs(self):
            return TensorDict({"actor": torch.randn(n, 100), "critic": torch.randn(n, 120)},
                              batch_size=[n])

        def reset(self):
            return self.obs()

        def step(self, actions):
            return self.obs(), 1. - actions.square().mean(-1), probabilities.clone(), {}

    class SyntheticRunner(native.PpoRunner):
        def _policy_input_dims(self):
            return 100, 120

        def _create_wm_env(self, batch_size):
            assert batch_size == n
            return SyntheticWorldModelEnv()

    results = []
    for preserve_soft in (False, True):
        torch.manual_seed(42)
        runner = SyntheticRunner.__new__(SyntheticRunner)
        runner.cfg = native.PpoRunnerCfg(batch_size=n, num_steps_per_env=32,
                                        num_learning_epochs=1, num_mini_batches=2)
        runner._num_actions = 29
        runner.device = "cpu"
        runner._init_policy()
        storage = runner.alg.storage
        if preserve_soft:
            # Diagnostic alternative only; no installed/vendor source is edited.
            storage.dones = torch.zeros_like(storage.dones, dtype=torch.float32)
        initial = [p.detach().clone() for p in runner.alg.actor.parameters()]
        losses = runner.imagine()
        stored = storage.dones[0, :, 0]
        continuation = 1. - stored.float()
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
        assert any(not torch.equal(before, after)
                   for before, after in zip(initial, runner.alg.actor.parameters(), strict=True))
        assert all(torch.equal(storage.dones[i, :, 0], stored) for i in range(32))
        results.append({
            "mode": "diagnostic_float_storage" if preserve_soft else "released_uint8_storage",
            "storage_dtype": str(storage.dones.dtype),
            "stored_termination": stored.tolist(),
            "gae_continuation": continuation.tolist(),
            "native_ppo_update_losses": losses,
            "actor_updated_and_finite": True,
            "rollout_steps": 32,
            "parallel_synthetic_envs": n,
        })
    native_stored = torch.tensor(results[0]["stored_termination"])
    soft_stored = torch.tensor(results[1]["stored_termination"])
    fractional = (probabilities > 0.) & (probabilities < 1.)
    assert fractional.any() and torch.equal(native_stored[fractional], torch.zeros_like(native_stored[fractional]))
    assert torch.equal(soft_stored, probabilities)
    return {
        "status": "termination_boundary_diagnostic_completed",
        "platform": "CPU; synthetic observations/rewards; actual released termination and PPO code",
        "torch_version": torch.__version__,
        "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
        "seed_per_arm": 42,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_ppo_actor_hidden_dims": [512, 256, 128],
        "native_ppo_critic_hidden_dims": [512, 256, 128],
        "source_files": {"ppo_runner": file_record(native), "world_model_env": file_record(wm_module),
                         "rsl_ppo": file_record(rsl_ppo), "rsl_rollout_storage": file_record(rsl_storage)},
        "thresholds": {"height": .25, "tilt": .8, "softness": .05},
        "reference_height_errors": height_errors.tolist(),
        "native_soft_termination_probabilities": probabilities.tolist(),
        "fractional_probabilities_truncated_to_zero": int(fractional.sum()),
        "arms": results,
        "conclusion": "Released PPO-WM forwards continuous soft termination to RSL uint8 storage; fractional probabilities become zero before GAE.",
        "alternative_is_only_a_diagnostic": True,
        "author_intended_ppo_termination_confirmed": False,
        "paper_equivalent_ppo_wm_implemented": False,
        "g1_physics_executed": False,
        "cuda_executed": False,
        "policy_quality_evaluated": False,
        "vendor_files_modified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run_probe()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print("PPO_WM_TERMINATION_CONTRACT_PASS " + json.dumps({
        "fractional_probabilities_truncated_to_zero": result["fractional_probabilities_truncated_to_zero"],
        "real_native_ppo_updates": len(result["arms"]), "g1_physics_executed": False}))
