"""Read an immutable native model snapshot; never claim full training resume."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read_snapshot(manifest_path, expected_collections, runner_config, corpus):
    import torch
    path=Path(manifest_path).resolve()
    manifest=json.loads(path.read_bytes())
    if (manifest['status']!='model_snapshot_preserved'
            or manifest['collection_steps']!=expected_collections
            or expected_collections<=0):
        raise ValueError('Unexpected native source snapshot step/status')
    source_config=path.parents[2]/'resolved-config.json'
    original=json.loads(source_config.read_bytes())
    if original['corpus']['name']!=corpus or original['training_profile']!='paper-a5-single-gpu':
        raise ValueError('Warm-start corpus/training profile mismatch')
    actual=json.loads(json.dumps(runner_config,default=str))
    for key in ('actor_obs','history_length','dynamics_hidden_dims','actor_hidden_dims',
                'critic_hidden_dims','latent_dim','wm','policy','replay_ratio','buffer_size'):
        if actual[key]!=original['runner'][key]:
            raise ValueError('Warm-start learning configuration mismatch: '+key)
    files=manifest['files']
    if len(files)!=3 or {item['file'] for item in files}!={'world_model.pt','actor.pt','critic.pt'}:
        raise ValueError('All three native model files are required')
    payloads={}

    def finite(value):
        if isinstance(value,torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all().item():
                raise ValueError('Nonfinite warm-start tensor')
        elif isinstance(value,dict):
            for child in value.values():finite(child)
        elif isinstance(value,(tuple,list)):
            for child in value:finite(child)

    for item in files:
        model=path.parent/item['file']
        if model.stat().st_size!=item['bytes'] or digest(model)!=item['sha256']:
            raise ValueError('Warm-start model bytes/hash mismatch')
        payload=torch.load(model,map_location='cpu',weights_only=True)
        finite(payload);payloads[item['file']]=payload
    if set(payloads['world_model.pt'])!={'dynamics','sim_context'}:
        raise ValueError('Expected the native dynamics/context bundle')
    evidence=dict(kind='native_model_weight_warm_start',source_manifest=str(path),
        source_manifest_sha256=digest(path),source_config_sha256=digest(source_config),
        inherited_collection_steps=expected_collections,learning_rate_offset=expected_collections,
        source_files=files,optimizer_state_restored=False,replay_buffer_restored=False,
        sampler_state_restored=False,simulator_state_restored=False,RNG_state_restored=False,
        exact_training_continuation=False,normalization_buffers_restored=True)
    return payloads,evidence


def restore_snapshot(runner,payloads):
    import torch
    runner._wm.load_bundle(payloads['world_model.pt'])
    runner.policy_learner.actor.load_state_dict(payloads['actor.pt'],strict=True)
    runner.policy_learner.critic.load_state_dict(payloads['critic.pt'],strict=True)
    pairs=[(runner._dynamics,payloads['world_model.pt']['dynamics']),
           (runner._sim_context,payloads['world_model.pt']['sim_context']),
           (runner.policy_learner.actor,payloads['actor.pt']),
           (runner.policy_learner.critic,payloads['critic.pt'])]
    for module,expected in pairs:
        actual=module.state_dict()
        if set(actual)!=set(expected) or not all(torch.equal(actual[key].detach().cpu(),value)
                                               for key,value in expected.items()):
            raise ValueError('Native warm-start live state differs from source')
    return dict(all_four_native_state_dictionaries_equal_source=True,
                all_learned_weights_and_normalization_buffers_read_back=True)


def learned_parameter_hashes(runner):
    import torch
    modules=dict(world_model=runner._dynamics,actor=runner.policy_learner.actor,
                 critic=runner.policy_learner.critic)
    result={}
    for label,module in modules.items():
        hasher=hashlib.sha256()
        for name,parameter in module.named_parameters():
            hasher.update((name+str(parameter.dtype)+str(tuple(parameter.shape))).encode())
            raw=parameter.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes()
            hasher.update(raw)
        result[label]=hasher.hexdigest()
    return result
