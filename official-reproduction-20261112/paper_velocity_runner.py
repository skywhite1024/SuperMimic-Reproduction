"""Local mixed-interface runner, extending the frozen author runner externally."""

import torch

from supermimic.learners.policy_learner import ActorCriticPolicyLearner, ActorCriticPolicyLearnerCfg
from supermimic.replay_buffer import FieldSpec
from supermimic.runners.reset_sampling import ReturnSampler
from supermimic.runners.supermimic_runner import SuperMimicRunner
from supermimic.tasks.task_spec import TaskSpec, build_actor_obs_td, make_motion_task

from paper_profile import install_paper_runner
from paper_velocity_actor import InterfaceCritic, PaperVelocityActor
from paper_velocity_contract import ReferenceVelocityCommands, VelocityCommandSchedule
from paper_velocity_runtime import MixedVelocityWorldModelEnv


def _scheduled_velocity_only(*_):
    raise RuntimeError("Use the local velocity runner's scheduled commands, not native goal extraction")


class MotionOnlyReturnSampler:
    """Derive shared reference reset weights from the motion interface only.

    This explicit reset-distribution choice is not specified in Appendix G.
    It prevents unrelated velocity rewards from scoring motion-reference bins.
    """

    def __init__(self, sampler, count):
        self.sampler, self.count = sampler, count

    def record(self, motion_times, rewards, terminated, clip_end, truncated):
        return self.sampler.record(*(value[:self.count] for value in
            (motion_times, rewards, terminated, clip_end, truncated)))


def make_velocity_runner(mode):
    if mode not in ("rl", "distill"):
        raise ValueError("Velocity mode must be rl or distill")

    class VelocityRunner(SuperMimicRunner):
        def __init__(self, *args, **kwargs):
            self.velocity_mode = mode
            super().__init__(*args, **kwargs)

        def _buffer_schema(self):
            return super()._buffer_schema() | {
                "velocity_command": FieldSpec(3),
                "velocity_held": FieldSpec(3),
                "velocity_reference": FieldSpec(dtype=torch.bool),
                "velocity_age": FieldSpec(dtype=torch.long),
                "velocity_interface": FieldSpec(dtype=torch.bool),
            }

        def _init_policy(self):
            if (self.cfg.actor_obs != "proprio" or self.cfg.history_length != 10
                    or self.cfg.representation != "gencoord"
                    or self.cfg.slot_mix != {"motion": .5, "velocity": .5}):
                raise ValueError("The mixed paper adapter requires proprio/history10/gencoord and equal interfaces")
            if self.env.num_envs % 2 or self.cfg.policy.batch_size % 2:
                raise ValueError("Even real and imagined environment counts are required")
            if not isinstance(self.cfg.policy, ActorCriticPolicyLearnerCfg):
                raise ValueError("Independent interface critics require actor-critic training")
            tasks = [make_motion_task(self._representation, self._num_goals),
                     TaskSpec("velocity", 3, _scheduled_velocity_only)]
            self._active_task_specs = tasks
            actor = PaperVelocityActor(
                self._representation.proprio_obs_dim(self._num_actions), self._num_actions,
                tasks, self.cfg.slot_mix, history_length=self.cfg.history_length,
                latent_dim=self.cfg.latent_dim, motion_hidden_dims=self.cfg.actor_hidden_dims,
                latent_bc_weight=self.cfg.latent_bc_weight).to(self.device)
            critic = InterfaceCritic(self._representation.rep_state_dim, self._num_goals,
                self._sim_context.output_dim, hidden_dims=self.cfg.critic_hidden_dims).to(self.device)
            self._reference_velocity = ReferenceVelocityCommands(self.motion)
            self._sim_commands = VelocityCommandSchedule(self.env.num_envs, mode=mode, device=self.device)
            self._previous_dones = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
            self._sim_velocity_rows = torch.arange(self.env.num_envs, device=self.device) >= self.env.num_envs // 2
            self.env.unwrapped._paper_velocity_command = torch.zeros(self.env.num_envs, 3, device=self.device)
            self.wm_env = MixedVelocityWorldModelEnv(
                dynamics=self._dynamics, representation=self._representation, env=self.env,
                batch_size=self.cfg.policy.batch_size, replay_buffer=self.replay_buffer,
                task_specs=tasks, history_length=self.cfg.history_length,
                termination_softness=self.cfg.termination_softness, device=self.device,
                velocity_mode=mode)
            if torch.device(self.device).type == "cuda":
                self.wm_env.step = torch.compile(self.wm_env.step)
                self.wm_env.reset = torch.compile(self.wm_env.reset)
            self._configure_reset_sampling()
            self.policy_learner = ActorCriticPolicyLearner(
                actor, critic, self.wm_env, self.cfg.policy, device=self.device)
            print(f"VELOCITY_INTERFACES mode={mode} real_half={self.env.num_envs // 2} "
                  f"imagined_half={self.cfg.policy.batch_size // 2} critics=2", flush=True)

        def _configure_reset_sampling(self):
            expected = "external" if self.cfg.sampling_mode == "return" else self.cfg.sampling_mode
            if self.motion_cmd.cfg.sampling_mode != expected:
                raise RuntimeError("Motion reset sampler and simulator command mode disagree")
            self._sampler = None
            if self.cfg.sampling_mode == "return":
                count = self.env.num_envs // 2
                native = ReturnSampler(cfg=self.cfg.return_sampler,
                    bin_count=self.motion_cmd.bin_count, frame_count=self.motion.time_step_total,
                    num_envs=count, max_step_reward=self.wm_env.motion_reward_ceiling, device=self.device)
                self._sampler = MotionOnlyReturnSampler(native, count)

        def _build_sim_obs_td(self, rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx):
            sensor = self._representation.compute_actor_sensor(rep_state, sim_ctx,
                aux_state=aux_state, sim_actor_obs=sim_actor_obs, noisy=True)
            _, motion_goal = self._representation.compute_wm_goal(self.motion, motion_times, aux_state)
            reference = self._reference_velocity.lookup(motion_times, validate=False)
            command, reference_mask = self._sim_commands.command(reference, self._previous_dones)
            self.env.unwrapped._paper_velocity_command = command
            self._velocity_transition = {
                "velocity_command": command,
                "velocity_held": self._sim_commands.held,
                "velocity_reference": reference_mask,
                "velocity_age": self._sim_commands.age - 1,
                "velocity_interface": self._sim_velocity_rows,
            }
            obs = build_actor_obs_td({"motion": motion_goal, "velocity": command},
                                      self._collection_history(sensor))
            obs["velocity_reference_mask"] = reference_mask
            return obs

        def _step_and_store(self, actions, curr_states, actor_td=None, extra_fields=None):
            extra = dict(extra_fields or {}) | self._velocity_transition
            result = super()._step_and_store(actions, curr_states, actor_td=actor_td, extra_fields=extra)
            self._previous_dones = result[2].bool().clone()
            return result

        def run(self):
            # The native exporter reconstructs a command-only velocity encoder.
            # Its ONNX graph cannot represent this history-conditioned adapter.
            import supermimic.runners.supermimic_runner as module
            original = module.export_actor_as_onnx
            def skip_export(**_):
                print("VELOCITY_ONNX_EXPORT_SKIPPED local history-conditioned actor", flush=True)
                return []
            module.export_actor_as_onnx = skip_export
            try:
                return super().run()
            finally:
                module.export_actor_as_onnx = original

    return install_paper_runner(VelocityRunner)
