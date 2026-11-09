"""Exercise the complete new batch loop with a fake backend, not simulation."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
sys.path.insert(0, str(WORKSPACE/"publication/SuperMimic-Reproduction/experiments/sports-video"))
import numpy as np
import torch
import sports_runtime
import evaluate_batch as batch
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY as STATE, SUPERMIMIC_TRANSITION_KEY as TRANS
from unisupermimic.tasks.motion_tracking.g1.supermimic import evaluate as upstream


class FakeEnvironment:
    def __init__(self):
        motion = SimpleNamespace(names=["a", "b", "c"], clip_len=torch.tensor([6, 5, 4]),
                                 clip_start=torch.tensor([0, 6, 11]), clip_end=torch.tensor([5, 10, 14]),
                                 clip_id=torch.tensor([0]*6+[1]*5+[2]*4))
        self.supermimic_mdp = SimpleNamespace(motion=motion,
            obs=lambda *args: torch.zeros((3, 101)), goal=lambda *args: torch.zeros((3, 73)),
            termination=lambda state, frames: state[0][:, 2] < 0)
        self._supermimic_command = SimpleNamespace(f=torch.zeros(3, dtype=torch.long), max_episode_steps=10**9)
        self.backend = SimpleNamespace(close=lambda: None)
        self.s = torch.zeros((3, 71)); self.s[:, 2] = .7; self.s[:, 3] = 1
        self.aprev = torch.zeros((3, 29)); self.calls = 0; self.initial_resets = 0

    def info(self):
        return {"s": self.s.clone(), "f": self._supermimic_command.f.clone(),
                "a_prev": self.aprev.clone(), "xi": torch.zeros((3, 1))}

    def reset_to_frames(self, frames):
        self.initial_resets += 1
        self._supermimic_command.f.copy_(frames)
        return {}, {STATE: self.info()}

    def reset(self):
        raise AssertionError("Unexpected reset")

    def step(self, action):
        prior = self._supermimic_command.f.clone()
        motion = self.supermimic_mdp.motion
        assert bool((prior+1 <= motion.clip_end[motion.clip_id[prior]]).all())
        if self.calls >= 2:
            assert torch.equal(action[0], torch.zeros(29)), "Inactive action must be zero"
        self.calls += 1
        self.s[:, 0] += .01
        if self.calls >= 2:
            self.s[0, 2] = -.3
        if self.calls >= 3:
            self.s[2, 2] = -.3
        self._supermimic_command.f.add_(1)
        self.aprev.copy_(action)
        tr = {"s1": self.s.clone(), "f": prior, "ep_step": torch.full((3,), self.calls-1),
              "r": torch.full((3,), .05), "term": torch.zeros(3, dtype=torch.bool),
              "done": torch.zeros(3, dtype=torch.bool)}
        return SimpleNamespace(info={STATE: self.info(), TRANS: tr})

    def close(self):
        pass


def fake_snapshot(mdp, current):
    state = current["s"].numpy().copy()
    return {"state": state, "reference": np.zeros((3, 41)), "root_xy_error": state[:, :2],
            "root_error_m": np.linalg.norm(state[:, :3], axis=1), "yaw_error_rad": np.zeros(3),
            "root_orientation_error_rad": np.zeros(3), "body_global_mean_m": np.zeros(3),
            "body_local_mean_m": np.zeros(3)}


env = FakeEnvironment()
sources = [{"path": Path(name+".csv"), "sha256": "mock", "group": "old", "fps": 30.} for name in ("a", "b", "c")]
saved = {"args": {"hist": 2, "goal_frames": 1}}
with tempfile.TemporaryDirectory() as folder, patch.object(sports_runtime, "make_env", return_value=env), \
     patch.object(upstream, "load_policy", return_value=lambda history, goal: torch.ones((3, 29))), \
     patch.object(batch, "snapshot_batch", side_effect=fake_snapshot):
    target = Path(folder)/"mock"
    result = batch.run_batch(sources, saved, device="cpu", feedback="F0", output=target)
    assert result["batch_error"] is None, result["batch_error"]
    assert env.initial_resets == 1 and env.calls == 4
    assert [row["observed_steps"] for row in result["clips"]] == [2, 4, 3]
    assert [row["status"] for row in result["clips"]] == ["native_hard_failure", "completed", "native_hard_failure"]
    assert [row["post_stop_backend_steps_discarded"] for row in result["clips"]] == [2, 0, 1]
    for i, count in enumerate((2, 4, 3)):
        with np.load(target/f"clip-{i:04d}-{sources[i]['path'].stem}"/"trajectory.npz") as trace:
            assert len(trace["state"]) == count+1
            assert len(trace["reward"]) == count
            assert np.all(trace["action_raw"] == 1), "Export must contain scored actions only"
    receipt = {"status": "pass", "scope": "complete evaluator loop with mocked physics/model/snapshot",
               "worlds": 3, "backend_steps": env.calls, "observed_steps": [2, 4, 3],
               "post_stop_backend_steps_discarded": [2, 0, 1], "initial_resets": env.initial_resets,
               "cuda_executed": False, "mujoco_executed": False, "policy_executed": False,
               "native_batch_equivalence_verified": False}
    (HERE/"mock-loop-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt))
