"""CPU contract tests, including native G1 physics; no cloud or training submission.

Run: python -m unittest discover -s experiments/sports-video -p test_sports_runtime.py -v
Set SPORTS_TEST_PARENT to additionally check a real full 99D checkpoint.
"""
import copy
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import torch
import sports_runtime as runtime
from sports_runtime import (UNI, XYObservationMDP, extra_columns, initial_policy_identity,
    install_f0_column_freeze, load_manifest, make_env, migrate_checkpoint, policy_from_checkpoint)
from unisupermimic.rl.algos.supermimic.geometry import so3_exp, unflatten_state
from unisupermimic.rl.algos.supermimic.networks import Policy, WorldModel, Critic
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import default_joint_pos_list


def tiny_parent():
    torch.manual_seed(123)
    args = {"hist": 3, "z_dim": 4, "goal_frames": 2, "policy_hidden": [16, 8], "amp": "none"}
    policy = Policy(99, 146, hist=3, z_dim=4, hidden=(16, 8))
    wm, critic = WorldModel(hidden=(16,)), Critic(20, hidden=(8,))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3, betas=(.9, .95))
    # Populate nonzero moments and counters before migration.
    policy(torch.randn(4, 3, 99), torch.randn(4, 146)).square().sum().backward()
    optimizer.step()
    return {"pi": copy.deepcopy(policy.state_dict()), "wm": wm.state_dict(), "critic": critic.state_dict(),
            "opt_pi": optimizer.state_dict(), "opt_wm": torch.optim.Adam(wm.parameters()).state_dict(),
            "opt_v": torch.optim.Adam(critic.parameters()).state_dict(), "args": args,
            "iteration": 16, "sim_steps": 1000, "wall_time": 1.}


def write_standing(path, frames=121):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for _ in range(frames):
            writer.writerow([0, 0, .76, 0, 0, 0, 1, *default_joint_pos_list()])


class SportsRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.csv = cls.directory / "standing.csv"
        write_standing(cls.csv)
        cls.kin = G1Kinematics(str(UNI / "src/unisupermimic/assets/robots/g1/supermimic.xml"), "cpu")
        cls.motion = MotionLib([str(cls.csv)], cls.kin, "cpu", src_fps=30.)
        cls.default = torch.tensor(default_joint_pos_list())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_observation_is_reference_yaw_only_and_native_contract_is_inherited(self):
        base = MDP(self.kin, self.motion, self.default)
        f0 = XYObservationMDP(self.kin, self.motion, self.default, feedback="F0")
        f1 = XYObservationMDP(self.kin, self.motion, self.default, feedback="F1")
        frames = torch.tensor([2, 5])
        state = tuple(v.clone() for v in self.motion.ref_state(frames))
        state[0][:, :2] += torch.tensor([[1., 2.], [-2., 1.]])
        state[1][:] = so3_exp(torch.tensor([[0., 0., math.pi/2], [.1, -.2, -.5]]))
        previous, context = torch.zeros(2, 29), torch.zeros(2, 33)
        obs0 = f0.obs(state, previous, context, frames)
        obs1 = f1.obs(state, previous, context, frames)
        self.assertEqual(obs0.shape, (2, 101))
        torch.testing.assert_close(obs0[:, :99], base.obs(state, previous, context, frames), rtol=0, atol=0)
        torch.testing.assert_close(obs1[:, :99], obs0[:, :99], rtol=0, atol=0)
        torch.testing.assert_close(obs0[:, -2:], torch.zeros(2, 2), rtol=0, atol=0)
        # Reference yaw is zero. Robot yaw differs, so robot-frame XY would fail.
        torch.testing.assert_close(obs1[:, -2:], torch.tensor([[-1., -2.], [2., -1.]]), rtol=0, atol=0)
        for method in ("reward", "termination", "soft_termination", "critic_features"):
            self.assertIs(getattr(XYObservationMDP, method), getattr(MDP, method))
        for mdp in (f0, f1):
            torch.testing.assert_close(mdp.reward(state, frames, previous, previous),
                                       base.reward(state, frames, previous, previous), rtol=0, atol=0)
            torch.testing.assert_close(mdp.termination(state, frames), base.termination(state, frames))
            self.assertEqual(mdp.max_step_reward, base.max_step_reward)
            self.assertTrue(torch.equal(mdp.obs_noise[-2:], torch.zeros(2)))

    def test_migration_preserves_parent_moments_and_initial_outputs(self):
        parent = tiny_parent()
        saved_parent = copy.deepcopy(parent)
        variants = [migrate_checkpoint(parent, feedback) for feedback in ("F0", "F1")]
        for migrated in variants:
            self.assertLessEqual(initial_policy_identity(parent, migrated)["max_abs_error"], 2e-5)
            for name in ("wm", "critic"):
                for key in parent[name]:
                    self.assertTrue(torch.equal(parent[name][key], migrated[name][key]))
            for key in ("mean", "var", "count"):
                actual = migrated["pi"][f"obs_norm.{key}"]
                if key != "count":
                    actual = actual[:99]
                self.assertTrue(torch.equal(actual, parent["pi"][f"obs_norm.{key}"]))
            old_policy = policy_from_checkpoint(parent)
            names = [name for name, _ in old_policy.named_parameters()]
            decoder_id = parent["opt_pi"]["param_groups"][0]["params"][names.index("dec.0.weight")]
            original_state, new_state = parent["opt_pi"]["state"][decoder_id], migrated["opt_pi"]["state"][decoder_id]
            self.assertTrue(torch.equal(original_state["step"], new_state["step"]))
            for slot in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(original_state[slot][:, :4], new_state[slot][:, :4]))
                self.assertTrue(torch.equal(original_state[slot][:, 4:].reshape(-1, 3, 99),
                                            new_state[slot][:, 4:].reshape(-1, 3, 101)[:, :, :99]))
                self.assertEqual(float(new_state[slot][:, extra_columns(3, 4)].abs().max()), 0.)
        for key in variants[0]["pi"]:
            self.assertTrue(torch.equal(variants[0]["pi"][key], variants[1]["pi"][key]))
        for key in parent["pi"]:
            self.assertTrue(torch.equal(parent["pi"][key], saved_parent["pi"][key]))

    def test_f0_columns_remain_zero_after_adam_while_f1_can_learn(self):
        for feedback in ("F0", "F1"):
            saved = migrate_checkpoint(tiny_parent(), feedback)
            policy = policy_from_checkpoint(saved)
            policy.load_state_dict(saved["pi"])
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3, betas=(.9, .95))
            optimizer.load_state_dict(saved["opt_pi"])
            hooks = install_f0_column_freeze(policy, optimizer) if feedback == "F0" else ()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                # Deliberately nonzero new channels stress the F0 weight freeze.
                obs = torch.randn(4, 3, 101)
                (policy(obs, torch.randn(4, 146))-1).square().mean().backward()
                optimizer.step()
            cols = extra_columns(3, 4)
            maximum = float(policy.dec[0].weight[:, cols].detach().abs().max())
            self.assertEqual(maximum == 0, feedback == "F0")
            if feedback == "F0":
                for slot in ("exp_avg", "exp_avg_sq"):
                    self.assertEqual(float(optimizer.state[policy.dec[0].weight][slot][:, cols].abs().max()), 0.)
            for hook in hooks:
                hook.remove()

    def test_manifest_rejects_mixed_fps_and_bad_hash(self):
        other = self.directory / "other.csv"
        write_standing(other)
        rows = [{"path": p.name, "fps": fps, "group": group, "sha256": runtime.sha(p)}
                for p, fps, group in ((self.csv, 30, "old"), (other, 50, "new"))]
        path = self.directory / "manifest.json"
        path.write_text(json.dumps({"clips": rows}))
        with self.assertRaisesRegex(ValueError, "Mixed source FPS"):
            load_manifest(path)
        rows[1]["fps"] = 30
        rows[1]["sha256"] = "0"*64
        path.write_text(json.dumps({"clips": rows}))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_manifest(path)

    def test_native_cpu_physics_f0_f1_initial_rollouts_match(self):
        parent = tiny_parent()
        trajectories, rewards = [], []
        for feedback in ("F0", "F1"):
            env = make_env([self.csv], "cpu", 1, 13, feedback, 30., evaluation=True)
            try:
                saved = migrate_checkpoint(parent, feedback)
                policy = policy_from_checkpoint(saved)
                policy.load_state_dict(saved["pi"]); policy.eval()
                _, info = env.reset_to_frames(env.supermimic_mdp.motion.clip_start.clone())
                current = info[SUPERMIMIC_STATE_KEY]
                history, states, reward_rows = None, [], []
                with torch.inference_mode():
                    for _ in range(8):
                        mdp = env.supermimic_mdp
                        obs = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"])
                        history = obs[:, None].repeat(1, 3, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
                        step = env.step(policy(history, mdp.goal(current["f"], 2)))
                        current = step.info[SUPERMIMIC_STATE_KEY]
                        states.append(current["s"].clone()); reward_rows.append(step.reward.clone())
                trajectories.append(torch.stack(states)); rewards.append(torch.stack(reward_rows))
            finally:
                env.close(); env.backend.close()
        self.assertTrue(torch.isfinite(trajectories[0]).all())
        torch.testing.assert_close(trajectories[0], trajectories[1], rtol=0, atol=1e-7)
        torch.testing.assert_close(rewards[0], rewards[1], rtol=0, atol=1e-7)

    @unittest.skipUnless(os.environ.get("SPORTS_TEST_PARENT"), "Set SPORTS_TEST_PARENT for the real parent check")
    def test_real_parent_identity_and_world_model_dimensions(self):
        path = Path(os.environ["SPORTS_TEST_PARENT"])
        self.assertEqual(runtime.sha(path), runtime.PARENT_SHA256)
        parent = torch.load(path, map_location="cpu", weights_only=True)
        for feedback in ("F0", "F1"):
            migrated = migrate_checkpoint(parent, feedback)
            self.assertEqual(migrated["pi"]["dec.0.weight"].shape, (2048, 1042))
            self.assertEqual(migrated["wm"]["in_norm.mean"].numel(), 133)
            self.assertEqual(migrated["wm"]["out_norm.mean"].numel(), 70)
            self.assertLessEqual(initial_policy_identity(parent, migrated)["max_abs_error"], 2e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
