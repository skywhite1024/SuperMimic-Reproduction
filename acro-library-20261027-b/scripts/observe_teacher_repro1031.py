"""Read-only first-policy-call observation around the unchanged frozen evaluator."""
import hashlib, importlib.metadata as metadata, json, os, platform, subprocess, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
import evaluate_library as evaluator
import evaluate_qualified_expert_v3 as frozen

def digest(value):
    tensor = value.detach().contiguous().cpu()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}

def main():
    mode = sys.argv[1]
    assert mode in [f'teacher8808-repro1031c-r{i}' for i in range(3)]
    native = evaluator.load_policy
    def load(env, saved, device=None):
        policy = native(env, saved, device)
        assert not policy.training and not policy.obs_norm.training
        record = {'mode': mode, 'seed': 0, 'read_only_hooks': True,
                  'checkpoint_args': {k: saved['args'].get(k) for k in ['amp','hist','goal_frames','z_dim','policy_hidden']},
                  'normalization': {k: digest(v) for k,v in saved['pi'].items() if k.startswith('obs_norm.')},
                  'python': platform.python_version(), 'torch': torch.__version__, 'CUDA_runtime': torch.version.cuda,
                  'GPU': torch.cuda.get_device_name(0), 'compute_capability': list(torch.cuda.get_device_capability(0)),
                  'float32_matmul_precision': torch.get_float32_matmul_precision(),
                  'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                  'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                  'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
                  'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
                  'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                  'packages': {}}
        for name in ['mujoco','mujoco-warp','warp-lang','unilab','hydra-core']:
            try: record['packages'][name] = metadata.version(name)
            except metadata.PackageNotFoundError: record['packages'][name] = None
        record['driver_version'] = subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip()
        captured = {}
        def before(module, inputs):
            if captured: return
            captured['history'] = inputs[0].detach().clone()
            captured['goal'] = inputs[1].detach().clone()
            record['initial_history'] = digest(captured['history'])
            record['initial_goal'] = digest(captured['goal'])
        def after(module, inputs, output):
            if 'first_raw_action' in record: return
            record['first_raw_action'] = digest(output)
            folder = ROOT/'runs'/mode
            np.savez_compressed(folder/'initial-policy-inputs.npz',
                                history=captured['history'].float().cpu().numpy(),
                                goal=captured['goal'].float().cpu().numpy(),
                                raw_action=output.detach().float().cpu().numpy())
            (folder/'initial-policy-inputs.json').write_text(json.dumps(record,indent=2)+'\n')
        policy.register_forward_pre_hook(before)
        policy.register_forward_hook(after)
        return policy
    evaluator.load_policy = load
    checkpoint = ROOT/'policy/qualified-expert-8808.pt'
    sys.argv = [sys.argv[0], '--role','teacher','--suite','library','--reference-variant','raw',
                '--checkpoint',str(checkpoint),'--device','cuda','--mode',mode,
                '--group','new','--clips','88_08_stageii_unitree_g1']
    frozen.main()

if __name__ == '__main__': main()
