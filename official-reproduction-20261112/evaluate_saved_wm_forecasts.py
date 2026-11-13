"""Explicit 1..8-transition forecasts on every complete saved LAFAN40 window.

Uses two frozen native WMs, identical saved physical states/actions and nominal
context. This is an offline trajectory diagnostic, not the paper's online
1/7/40/264-hour dataset experiment. Raw per-group MSE arrays remain available.
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HORIZON=8
GROUPS=('root_pos','root_ori','joint_pos','joint_vel')
WIDTHS=np.array([3,9,29,29],dtype=np.float64)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def forecast(dynamics,representation,states,actions,context):
    """Eight actual actions; ninth unused pad matches the native window API."""
    from supermimic.learners.wm_learner import WorldModelLearner
    if states.shape[1:]!=(9,71) or actions.shape[1:]!=(8,29):
        raise ValueError('Require nine states and eight actual actions')
    if states.shape[0]!=actions.shape[0] or context.shape!=(states.shape[0],33):
        raise ValueError('Inconsistent forecast batch')
    padded=torch.cat((actions,torch.zeros_like(actions[:,:1])),dim=1)
    pred,aux=WorldModelLearner._rollout(SimpleNamespace(dynamics=dynamics),
        representation.represent(states[:,0]),padded,
        representation.extract_aux_state(states[:,0]),context)
    if pred.shape[1]!=HORIZON:raise RuntimeError('Native rollout transition convention changed')
    target=states[:,1:]
    pb=representation.loss_blocks(pred,aux)
    gb=representation.loss_blocks(representation.represent(target),representation.extract_aux_state(target))
    if tuple(pb)!=GROUPS or tuple(gb)!=GROUPS:raise ValueError('Review changed native loss groups')
    mse=torch.stack([(pb[key]-gb[key]).square().mean(dim=-1) for key in GROUPS],dim=-1)
    if not torch.isfinite(mse).all() or (mse<0).any():raise RuntimeError('Nonfinite forecast errors')
    return pred,aux,mse


def reduce_errors(path,nominals,clips):
    """Float64 reduction from saved raw group-MSE arrays, never training logs."""
    values=np.load(path,mmap_mode='r',allow_pickle=False)
    if values.dtype!=np.float32 or values.ndim!=3 or values.shape[1:]!=(8,4):
        raise ValueError('Unexpected full window error array')
    if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Invalid saved MSE')
    nominals=np.asarray(nominals,dtype=np.float64)
    weights=WIDTHS/nominals**2/WIDTHS.sum()
    means=values.mean(axis=0,dtype=np.float64)
    per_horizon=means@weights
    rows=[]
    cursor=0
    for clip in clips:
        count=clip['windows']
        if clip['offset']!=cursor or count!=max(0,clip['saved_steps']-8):
            raise ValueError('A window crosses a clip boundary or has incomplete future states')
        part=values[cursor:cursor+count].mean(axis=0,dtype=np.float64)
        rows.append(clip|{'group_mse_by_horizon':part.tolist(),
                          'normalized_state_mse_by_horizon':(part@weights).tolist()})
        cursor+=count
    if cursor!=len(values):raise ValueError('Missing or duplicate forecast windows')
    return dict(windows=cursor,group_names=list(GROUPS),group_widths=WIDTHS.astype(int).tolist(),
        nominal_errors=nominals.tolist(),group_mse_by_horizon=means.tolist(),
        normalized_state_mse_by_horizon=per_horizon.tolist(),
        normalized_state_rmse_by_horizon=np.sqrt(per_horizon).tolist(),
        mean_normalized_state_mse_over_horizons=float(per_horizon.mean()),
        equal_clip_normalized_state_mse_by_horizon=np.mean([r['normalized_state_mse_by_horizon'] for r in rows],axis=0).tolist(),
        clips=rows)


def numpy_groups(predicted,target):
    """Independent full-state error calculation for saved first/last windows."""
    def rotations(state):
        q=state[...,3:7].astype(np.float64)
        w,x,y,z=np.moveaxis(q,-1,0)
        scale=2/(q*q).sum(axis=-1)
        return np.stack([1-scale*(y*y+z*z),scale*(x*y-z*w),scale*(x*z+y*w),
            scale*(x*y+z*w),1-scale*(x*x+z*z),scale*(y*z-x*w),
            scale*(x*z-y*w),scale*(y*z+x*w),1-scale*(x*x+y*y)],axis=-1)*np.sqrt(.5)
    a,b=predicted.astype(np.float64),target.astype(np.float64)
    return np.stack([((a[...,:3]-b[...,:3])**2).mean(-1),
        ((rotations(a)-rotations(b))**2).mean(-1),
        ((a[...,7:36]-b[...,7:36])**2).mean(-1),
        ((a[...,42:71]-b[...,42:71])**2).mean(-1)],axis=-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-1k','run-16k','evaluation','trajectory-ledger','motions','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=4096)
    args=parser.parse_args()
    if args.batch_size!=4096:raise ValueError('This finite experiment fixes batch size4096')
    if args.output.exists():raise FileExistsError('Use a fresh output')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one allocated GPU required')
    os.environ.update(MUJOCO_GL='disable',WANDB_MODE='offline')
    import supermimic.tasks
    import supermimic.representations as representations
    from mjlab.tasks.registry import load_env_cfg
    from supermimic.runners.run_config import load_runner_cfg
    from supermimic.runners.supermimic_runner import build_world_model
    from supermimic.scripts.train_utils import create_env,setup_device
    device=setup_device('cuda:0')
    representations._compiled_assemble_goal=representations._assemble_goal
    args.output.mkdir(parents=True)
    evaluation=json.loads((args.evaluation/'result.json').read_bytes())
    protocol=json.loads((args.evaluation/'protocol.json').read_bytes())
    if evaluation['clips']!=40 or evaluation['manual_reset_version']!=2:raise ValueError('Require corrected full40 evaluation')
    with args.trajectory_ledger.open(encoding='utf-8',newline='') as stream:ledger=list(csv.DictReader(stream))
    if len(ledger)!=40 or [int(r['index']) for r in ledger]!=list(range(40)):raise ValueError('Incomplete saved trajectory ledger')
    records=evaluation['records']
    if [r['index'] for r in records]!=list(range(40)):raise ValueError('Incomplete evaluation identities')
    clips=[];total=0
    for record,item in zip(records,ledger,strict=True):
        if item['trajectory']!=record['trajectory'] or int(item['steps'])!=record['steps']:raise ValueError('Changed saved trajectory membership')
        windows=max(0,record['steps']-8)
        if not windows:raise ValueError('This full40 experiment expects at least one complete window per clip')
        clips.append(dict(index=record['index'],name=record['name'],success=record['success'],
            saved_steps=record['steps'],offset=total,windows=windows))
        total+=windows
    if total!=437276:raise ValueError('Changed complete-window population')
    models={};contexts={};configs={};bindings={};nominals={}
    cfg=load_env_cfg('SuperMimic-GenCoord-G1',play=True)
    cfg.seed=42;cfg.auto_reset=False
    cfg.commands['motion'].motion_file=str(args.motions)
    cfg.events['foot_friction'].params['ranges']=(1.,1.)
    cfg.events['base_com'].params['ranges']={axis:(0.,0.) for axis in range(3)}
    cfg.events['encoder_bias'].params['bias_range']=(0.,0.)
    for group in cfg.observations.values():group.enable_corruption=False
    robot=cfg.scene.entities['robot']
    robot.collisions=tuple(dataclasses.replace(c,friction=(1.,)) for c in robot.collisions)
    env=create_env(cfg,1,device)
    try:
        for label,run,steps in [('1k',args.run_1k,1001),('16k',args.run_16k,16001)]:
            r=json.loads((run/'result.json').read_bytes());c=json.loads((run/'resolved-config.json').read_bytes())
            if r['status']!='pilot_completed' or r['collection_steps']!=steps or r['training_profile']!='paper-a5-single-gpu':
                raise ValueError('Wrong completed training run')
            ck=next(x for x in r['checkpoints'] if Path(x['path']).name=='world_model.pt')
            path=Path(ck['path'])
            if path.stat().st_size!=ck['bytes'] or sha(path)!=ck['sha256']:raise ValueError('World model differs')
            rc=load_runner_cfg({'runner_cfg':c['runner']})
            if rc.representation!='gencoord' or rc.actor_obs!='proprio' or rc.history_length!=10:raise ValueError('Changed model representation')
            wm=build_world_model(rc,env,device)
            wm.load_bundle(torch.load(path,map_location=device,weights_only=True));wm.dynamics.eval();wm.dynamics.requires_grad_(False)
            if not all(torch.isfinite(v).all() for v in wm.dynamics.state_dict().values()):raise ValueError('Nonfinite model state')
            contexts[label]=wm.sim_context.read(env.unwrapped).detach()
            if contexts[label].shape!=(1,33) or not torch.isfinite(contexts[label]).all():raise ValueError('Invalid nominal context')
            models[label]=wm;configs[label]=c['runner']
            nominals[label]=[getattr(rc.wm,wm.representation.LOSS_WEIGHT_CONFIG[key]) for key in GROUPS]
            bindings[label]=dict(run=str(run),training_result_sha256=sha(run/'result.json'),resolved_config_sha256=sha(run/'resolved-config.json'),
                world_model=ck,normalized_context=contexts[label][0].cpu().tolist(),
                context_mean=wm.sim_context.mean.cpu().tolist(),context_scale=wm.sim_context.scale.cpu().tolist())
        excluded={'max_iterations','save_interval','run_name','log_interval'}
        critical_configs_equal={k:v for k,v in configs['1k'].items() if k not in excluded}=={k:v for k,v in configs['16k'].items() if k not in excluded}
        meta=dict(horizon_transitions=8,input_states_per_window=9,actual_actions_per_window=8,
            native_unused_action_padding=1,native_configured8state_window_scores_transitions=7,
            all_complete_windows=True,windows=total,trajectories=40,includes_failure_trajectory=True,
            model_bindings=bindings,critical_runner_configs_equal=critical_configs_equal,
            torch=torch.__version__,torch_cuda=torch.version.cuda,
            cuda_matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
            goal_assembly_backend='author eager function; forecast itself makes no goal assembly calls',
            evaluation_result_sha256=sha(args.evaluation/'result.json'),evaluation_protocol_sha256=sha(args.evaluation/'protocol.json'),
            trajectory_ledger_sha256=sha(args.trajectory_ledger),
            source_trajectory_policy='Same frozen16k LAFAN policy, correctedv2 physical rollouts for both WMs',
            context='New native nominal G1 physical context; normalized separately by each frozen WM bundle',
            metric='Native Bryson-weighted mean over70 channels: rootpos3, sqrt(1/2)rotmat9, jointpos29,jointvel29. Both per-transition and mean1..8 reported.',
            online_paper_measurement_reproduced=False,held_out_data_claimed=False,dataset_scale_sweep_completed=False,
            exact_paper_error_aggregation_confirmed=False,paper_results_reproduced=False)
        write(args.output/'protocol.json',meta)
        arrays={label:np.lib.format.open_memmap(args.output/f'{label}-group-mse.npy',mode='w+',dtype=np.float32,shape=(total,8,4)) for label in models}
        samples={label:{key:[] for key in ('predicted','target','initial','clip_index','start_step','global_window')} for label in models}
        with torch.no_grad():
            for clip,item,record in zip(clips,ledger,records,strict=True):
                path=args.evaluation/item['trajectory']
                if path.stat().st_size!=int(item['bytes']) or sha(path)!=item['sha256']:raise ValueError('Changed saved physical trajectory')
                with np.load(path,allow_pickle=False) as archive:
                    source={key:archive[key] for key in archive.files}
                states=torch.as_tensor(source['full_state'],device=device)
                actions=torch.as_tensor(source['actions'],device=device)
                if states.shape!=(clip['saved_steps'],71) or actions.shape!=(clip['saved_steps'],29):raise ValueError('Wrong physical trace shapes')
                if states.dtype!=torch.float32 or actions.dtype!=torch.float32 or not torch.isfinite(states).all() or not torch.isfinite(actions).all():raise ValueError('Invalid physical trace values')
                for label,wm in models.items():
                    for start in range(0,clip['windows'],args.batch_size):
                        stop=min(clip['windows'],start+args.batch_size)
                        ids=torch.arange(start,stop,device=device)
                        windows=states[ids[:,None]+torch.arange(9,device=device)[None,:]]
                        control=actions[ids[:,None]+torch.arange(8,device=device)[None,:]]
                        pred,aux,mse=forecast(wm.dynamics,wm.representation,windows,control,contexts[label].expand(stop-start,-1))
                        arrays[label][clip['offset']+start:clip['offset']+stop]=mse.cpu().numpy()
                        for chosen in sorted({0,clip['windows']-1}):
                            if start<=chosen<stop:
                                index=chosen-start
                                full=wm.representation.reconstruct_full_state(aux[index],pred[index]).cpu().numpy()
                                sample=samples[label]
                                for key,value in [('predicted',full),('target',windows[index,1:].cpu().numpy()),
                                    ('initial',windows[index,0].cpu().numpy()),('clip_index',clip['index']),
                                    ('start_step',chosen),('global_window',clip['offset']+chosen)]:sample[key].append(value)
                print('WM_FORECAST_CLIP_COMPLETED '+json.dumps(clip),flush=True)
        results={}
        for label,values in arrays.items():
            values.flush()
            path=args.output/f'{label}-group-mse.npy'
            sample={key:np.asarray(value) for key,value in samples[label].items()}
            np.savez_compressed(args.output/f'{label}-samples.npz',**sample)
            independent=numpy_groups(sample['predicted'],sample['target'])
            saved=values[sample['global_window']]
            if not np.allclose(independent,saved,atol=1e-7,rtol=2e-4):raise RuntimeError('Saved physical sample errors differ from independent NumPy calculation')
            results[label]=reduce_errors(path,nominals[label],clips)|dict(
                raw_group_array=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)),
                samples=dict(file=f'{label}-samples.npz',count=len(sample['clip_index']),
                    sha256=sha(args.output/f'{label}-samples.npz'),all_raw_groups_recomputed_numpy=True,
                    max_abs_group_difference=float(np.abs(independent-saved).max()),tolerance=dict(atol=1e-7,rtol=2e-4)))
        result=dict(status='complete_shared_trajectory_eight_transition_forecast_diagnostic',
            models=results,protocol_sha256=sha(args.output/'protocol.json'),gpu=torch.cuda.get_device_name(),
            worker_sha256=sha(Path(__file__)),all437276_windows_per_model_completed=True,
            raw_group_arrays_readback_verified=True,independent_first_and_last_window_checks_per_clip=True,
            float64_aggregation=True,critical_runner_configs_equal=critical_configs_equal,paper_results_reproduced=False)
        write(args.output/'result.json',result)
        print('WM_FORECAST_COMPLETED '+json.dumps({key:results[key]['normalized_state_mse_by_horizon'] for key in results}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
