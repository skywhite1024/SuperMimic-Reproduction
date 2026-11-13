"""Compare native synthetic proprio with native biased-joint observation on CPU.

No simulator or learned policy is run. The fake asset only supplies tensors to
the actual mjlab joint_pos_rel function. The native generalized-coordinate
representation and context denormalization are imported without modification.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config_path):
    import torch
    import mjlab.envs.mdp.observations as observations
    import supermimic.representations as representations
    import supermimic.sim_context as context_module

    config = json.loads(config_path.read_bytes())
    assert config['runner']['representation'] == 'gencoord' and config['runner']['actor_obs'] == 'proprio'
    lo, hi = config['environment']['events']['encoder_bias']['params']['bias_range']
    assert lo < 0 < hi
    assert config['environment']['observations']['actor']['terms']['joint_pos']['params']['biased'] is True
    n, joints = 3, 29
    default = torch.linspace(-.1, .1, joints)
    full = torch.zeros(n, 71)
    full[:, 2], full[:, 3] = .76, 1.
    full[:, 7:7+joints] = default + .03
    representation = representations.GenCoordRepresentation(joints, default)
    representation.actor_obs_mode = 'proprio'
    state = representation.represent(full)
    joint_state = representation.joint_state(state)
    bias = torch.stack([torch.zeros(joints), torch.full((joints,), hi / 2), torch.full((joints,), lo / 2)])
    spec = context_module.EncoderBiasSpec()
    spec.dim = joints  # Synthetic context shape, no binding to a simulator.
    context = context_module.SimContext([spec], [(lo, hi)] * joints)
    normalized = (bias - context.mean) / context.scale
    assert torch.equal(context.denormalize(normalized, context.find('encoder_bias')), bias)
    actual = representation.compute_actor_sensor(state, normalized, aux_state=full[:, :7],
        joint_state=joint_state, last_action=torch.zeros(n, joints), noisy=False)
    robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=default.expand(n, -1),
        joint_pos=full[:, 7:7+joints], joint_pos_biased=full[:, 7:7+joints] + bias))
    env = SimpleNamespace(scene={'robot': robot})
    real_joint = observations.joint_pos_rel(env, biased=True)
    errors = (actual[:, 6:6+joints] - real_joint).abs().amax(dim=1)
    assert actual.shape == (3, 93) and errors[0] == 0
    assert torch.allclose(errors[1:], bias[1:].abs().amax(dim=1), atol=1e-8, rtol=1e-6)
    assert torch.equal(actual[0], actual[1]) and torch.equal(actual[0], actual[2])
    return {'status':'native_encoder_bias_contract_diagnostic_completed',
        'platform':'CPU, synthetic full states and asset tensors; actual native observation/representation functions',
        'torch_version':torch.__version__, 'probe_sha256':digest(Path(__file__)),
        'frozen_training_config':{'path':str(config_path),'sha256':digest(config_path)},
        'native_sources':{name:{'path':inspect.getfile(module),'sha256':digest(Path(inspect.getfile(module)))}
            for name,module in [('observations',observations),('representations',representations),('sim_context',context_module)]},
        'joint_bias_randomization_range':[lo,hi], 'raw_bias_per_row':[0.,hi/2,lo/2],
        'maximum_joint_sensor_difference_per_row':errors.tolist(),
        'sensor_dimensions':list(actual.shape), 'zero_bias_control_matches':True,
        'nonzero_bias_does_not_change_native_synthetic_sensor':True,
        'conclusion':'GenCoord compute_wm_obs discards sim_ctx; native biased joint observation changes while synthesized proprio does not.',
        'policy_quality_effect_measured':False, 'g1_physics_executed':False,
        'cuda_executed':False, 'vendor_or_installed_files_modified':False,
        'running_training_modified':False,
        'related_unmerged_pr':'https://github.com/HybridRobotics/supermimic/pull/15'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.training_config)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print('GENCOORD_ENCODER_BIAS_DIAGNOSTIC_PASS '+json.dumps({
        'sensor_dimensions':result['sensor_dimensions'],
        'maximum_joint_sensor_difference_per_row':result['maximum_joint_sensor_difference_per_row'],
        'policy_quality_effect_measured':False}))
