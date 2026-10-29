"""Resume the unchanged learner into a new directory, with staged acceptance."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
import torch
from omegaconf import OmegaConf
from audit_common import file_hash, optimizer_hashes, state_hash, write_json, lr_at

OUT = Path(__file__).resolve().parent
BASE = Path(os.environ['SUPERMIMIC_BASE']).resolve()
POLICY = OUT / 'policy'
sys.path.insert(0, str(BASE))
from train_audited_v2 import AuditedLearner, snapshot
import unisupermimic.scripts.train_supermimic as entry
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner as OriginalLearner, _LOG_FIELDS
from unilab.training.seed import apply_configured_training_seed
from unilab.training.experiment import write_run_config_snapshot
from unilab.training import ensure_registries
from unilab.utils.device import get_default_device
from unilab.base.process_device import configure_backend_process_device
from unilab.base.config_adapter import create_env
from unisupermimic.sim2sim import extract_contract_snapshot


class FinishRun(Exception):
    pass


class ContinuationLearner(AuditedLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_requested = False
        self.resume_gate_passed = False
        self.evaluations = []
        self.finished_reason = 'iteration_limit'
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop_requested', True))

    def load(self, path):
        expected = json.loads((OUT / 'evidence/preflight.json').read_text())
        assert file_hash(path) == expected['parent_sha256']
        assert json.loads(json.dumps(asdict(self.cfg))) == {**expected['checkpoint_args'], 'iterations': 24000}
        OriginalLearner.load(self, path)
        self.audit_before = snapshot(self)
        self.audit_resume = {'checkpoint': str(path), 'sha256': file_hash(path)}
        actual = {name: state_hash(module.state_dict()) for name, module in [('wm', self.wm), ('pi', self.pi), ('critic', self.critic)]}
        actual.update(optimizer_hashes(self))
        assert actual == expected['state_hashes'], 'Models or full optimizer state did not restore exactly'
        assert self.iteration == 12000 and self.sim_steps == expected['sim_steps']
        assert self.buffer.size == 0
        self.loaded_identity = {'passed': True, 'iteration': self.iteration, 'sim_steps': self.sim_steps,
                                'state_hashes': actual, 'full_optimizer_moments_checked': True,
                                'replay_size_before_rewarm': self.buffer.size,
                                'first_iteration_lr': self.lr_at(self.iteration),
                                'rng_and_replay_bit_exact': False}
        write_json(OUT / 'evidence/loaded-identity.json', self.loaded_identity)
        print('CONTINUATION_LOAD_IDENTITY_PASS ' + json.dumps({'iteration': self.iteration, 'all_models_and_optimizer_moments_equal': True, 'first_lr': self.lr_at(self.iteration)}), flush=True)

    def collect(self, steps):
        if self.stop_requested:
            OriginalLearner.save(self, POLICY / 'emergency-full.pt', include_optimizers=True)
            OriginalLearner.save(self, POLICY / f'emergency-{self.iteration:06d}.pt', include_optimizers=False)
            self.finished_reason = 'finite_budget_or_operator_signal'
            raise FinishRun(self.finished_reason)
        if self.iteration >= 12005 and not self.resume_gate_passed:
            after = snapshot(self)
            assert after['iteration'] == 12005
            assert all(after[key]['finite'] and after[key]['sha256_parameters'] != self.audit_before[key]['sha256_parameters'] for key in ['wm', 'policy', 'critic'])
            assert all(after[key]['max'] > self.audit_before[key]['max'] for key in ['wm_optimizer', 'policy_optimizer', 'critic_optimizer'])
            assert after['sim_steps'] > self.audit_before['sim_steps']
            proof = {'passed': True, 'before': self.audit_before, 'after': after,
                     'replay_size_after_rewarm': self.rewarm_size, 'warmup_before_first_update_verified': True,
                     'first_lr_matches_absolute_iteration': True, 'optimizer_moments_load_identity': True}
            write_json(OUT / 'evidence/resume-acceptance.json', proof)
            print('CONTINUATION_RESUME_ACCEPTED ' + json.dumps({'iteration': after['iteration'], 'replay_size': self.rewarm_size, 'all_models_updated': True, 'all_optimizers_advanced': True}), flush=True)
            self.resume_gate_passed = True
        return super().collect(steps)

    def wm_update(self):
        if not hasattr(self, 'rewarm_size'):
            assert self.iteration == 12000
            self.rewarm_size = self.buffer.size
            assert self.rewarm_size >= max(self.cfg.warmup_steps, self.cfg.hist, self.cfg.wm_window)
            assert abs(self.lr_at(12000) - lr_at(12000)) < 1e-15
            for optimizer in [self.opt_wm, self.opt_pi, self.opt_v]:
                assert all(abs(group['lr'] - lr_at(12000)) < 1e-15 for group in optimizer.param_groups)
        return super().wm_update()

    def evaluate_checkpoint(self, checkpoint, suffix):
        destination = OUT / 'evaluations' / f'{self.iteration:06d}{suffix}.json'
        log = destination.with_suffix('.stdout.txt')
        destination.parent.mkdir(exist_ok=True)
        started = time.monotonic()
        with log.open('w') as stream:
            subprocess.run([sys.executable, '-u', str(BASE / 'evaluate_audited_v2.py'),
                            '--checkpoint', str(checkpoint), '--output', str(destination)],
                           cwd=BASE / 'uni', stdout=stream, stderr=subprocess.STDOUT, timeout=1200, check=True)
        result = json.loads(destination.read_text())
        assert len(result['per_clip']) == 40
        row = {'iteration': self.iteration, 'checkpoint': str(checkpoint),
               'checkpoint_sha256': result['checkpoint_sha256'], 'result': str(destination),
               'macro': result['norm_return'], 'frame_weighted': result['norm_return_framew'],
               'failed': result['failed'], 'n_failed': result['n_failed'],
               'wall_seconds': time.monotonic() - started, 'confirmation': suffix == '-confirm'}
        self.evaluations.append(row)
        write_json(OUT / 'evaluations/index.json', self.evaluations)
        print('CONTINUATION_EVAL ' + json.dumps(row), flush=True)
        return row

    def save(self, path=None, include_optimizers=True):
        result = super().save(path, include_optimizers)
        if path is not None and not include_optimizers and self.iteration in [16000, 20000, 24000]:
            assert self.resume_gate_passed
            assert all(abs(group['lr'] - lr_at(self.iteration - 1)) < 1e-15 for opt in [self.opt_wm, self.opt_pi, self.opt_v] for group in opt.param_groups)
            shutil.copyfile(POLICY / 'last.pt', POLICY / f'full_{self.iteration:06d}.pt')
            write_json(POLICY / f'update-state-{self.iteration:06d}.json', {'state': snapshot(self), 'optimizer_hashes': optimizer_hashes(self), 'lr': self.lr_at(self.iteration - 1)})
            row = self.evaluate_checkpoint(result, '')
            if row['macro'] >= 0.82 and row['n_failed'] == 0:
                confirm = self.evaluate_checkpoint(result, '-confirm')
                if confirm['macro'] >= 0.82 and confirm['n_failed'] == 0:
                    self.finished_reason = 'confirmed_community_target_met'
                    raise FinishRun(self.finished_reason)
        return result

    def train(self, resume=None):
        started = time.monotonic()
        summary = {}
        try:
            summary = OriginalLearner.train(self, resume)
        except FinishRun:
            pass
        after = snapshot(self)
        proof = {'before': self.audit_before, 'after': after, 'resume': self.audit_resume,
                 'resume_gate_passed': self.resume_gate_passed, 'finish_reason': self.finished_reason,
                 'entry_seconds_including_staged_evaluations': time.monotonic() - started,
                 'summary': summary, 'evaluations': self.evaluations,
                 'optimizer_hashes_after': optimizer_hashes(self),
                 'replay_and_rng_bit_exact': False, 'lr_decay_iters': self.cfg.lr_decay_iters,
                 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                 'transitions_by_clip': dict(zip(self.mdp.motion.names, self.clip_transition_counts.cpu().tolist()))}
        proof['passed'] = self.resume_gate_passed and after['iteration'] > 12000 and all(after[k]['finite'] for k in ['wm', 'policy', 'critic'])
        write_json(OUT / 'evidence/continuation-proof.json', proof)
        assert proof['passed']
        print('CONTINUATION_TRAIN_END ' + json.dumps({'iteration': after['iteration'], 'reason': self.finished_reason, 'passed': proof['passed']}), flush=True)
        return proof


def main():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert 'H100' in torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    assert not POLICY.exists(), 'Refusing to overwrite an existing continuation'
    parent_snapshot = json.loads((BASE / 'runs/baseline-seed0/run_config.json').read_text())
    cfg = OmegaConf.create(parent_snapshot['config'])
    cfg.algo.iterations = 24000
    cfg.algo.resume = True
    cfg.training.log_dir = str(POLICY)
    assert len(cfg.env.commands.motion.motion_files) == 40 and cfg.algo.num_envs == 4096
    seed = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    ensure_registries()
    device = get_default_device()
    bound = configure_backend_process_device('mjwarp', device)
    if bound is not None: device = bound
    env_override = entry.build_env_override(cfg, learner_device=str(device))
    POLICY.mkdir()
    write_run_config_snapshot(POLICY, run_metadata={**parent_snapshot['run'], 'seed': seed.to_dict()},
                              full_cfg=cfg, contract_snapshot=extract_contract_snapshot(cfg))
    assert json.loads((POLICY / 'run_config.json').read_text())['contract_snapshot'] == parent_snapshot['contract_snapshot']
    (POLICY / 'train_log.csv').write_text(','.join(_LOG_FIELDS) + '\n')
    write_json(OUT / 'provenance.json', {'parent': str(BASE / 'runs/baseline-seed0/last.pt'),
               'parent_sha256': file_hash(BASE / 'runs/baseline-seed0/last.pt'),
               'controlled_change': 'total iterations 12000 to 24000 only; same seed and frozen MDP/dependencies/evaluator',
               'control_fields': ['algo.iterations', 'algo.resume', 'training.log_dir'],
               'replay_rewarmed': True, 'rng_replay_bit_exact': False,
               'staged_eval_pauses_in_same_training_process': [16000, 20000, 24000],
               'target_is_community_acceptance_not_paper_lafan1_failure_claim': True})
    env = None
    try:
        env = create_env(cfg, num_envs=4096, env_cfg_override=env_override)
        learner = ContinuationLearner(env, entry.build_learner_config(cfg), POLICY, device=str(device), backend='mjwarp')
        assert learner.mdp.obs_dim == 99 and len(learner.mdp.motion.names) == 40
        learner.train(resume=BASE / 'runs/baseline-seed0/last.pt')
    finally:
        if env is not None:
            env.close(); env.backend.close()


if __name__ == '__main__':
    main()
