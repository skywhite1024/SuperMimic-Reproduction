"""Finite auxiliary analysis only; never invokes a training entrypoint."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import sys
import time

OUT=Path(__file__).resolve().parent
EXPERIMENT=OUT.parent
BASE=Path(os.environ['SUPERMIMIC_BASE']).resolve()
sys.path.insert(0,str(EXPERIMENT))
sys.path.insert(0,str(BASE))
from audit_common import file_hash,write_json
import run_gpu_v3 as monitor
monitor.ROOT=OUT
monitor.SOURCE=BASE/'uni'


def read(path):return json.loads(path.read_text())


def main():
    started=time.monotonic();deadline=started+3300
    (OUT/'evidence').mkdir()
    record={'status':'running','started_at':dt.datetime.now(dt.timezone.utc).isoformat(),
            'training_executed':False,'gpus':1,'program_budget_seconds':3300,
            'strict_replay_failure_preserved':True,'primary_evaluations_unchanged':True}
    write_json(OUT/'evidence/postprocess.json',record)
    try:
        from preflight import check
        checked=check()
        assert checked['passed']
        parent=read(EXPERIMENT/'experiment.json')
        proof=read(EXPERIMENT/'evidence/continuation-proof.json')
        selection=read(EXPERIMENT/'selection.json')
        assert parent['status']=='failed_or_interrupted' and parent['target_confirmed']
        assert proof['finish_reason']=='confirmed_community_target_met' and proof['after']['iteration']==16000
        checkpoint=EXPERIMENT/'policy/ckpt_016000.pt'
        digest=file_hash(checkpoint)
        assert digest=='44d68af11306f839fe1b4b37cd042f1eae7df116adaabd69bd8db593a16d334e'
        for suffix in ['', '-confirm']:
            row=read(EXPERIMENT/'evaluations'/('016000'+suffix+'.json'))
            assert row['checkpoint_sha256']==digest and len(row['per_clip'])==40 and row['norm_return']>=.82 and row['n_failed']==0
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
        write_json(OUT/'evidence/preflight.json',{'passed':True,'checkpoint_sha256':digest,
            'source_files_verified':checked['source_files_verified'],'asset_files_verified':checked['asset_files_verified'],
            'gpu':torch.cuda.get_device_name(0),'cpu_max':monitor.read_cgroup('cpu.max'),
            'memory_max':monitor.read_cgroup('memory.max'),
            'parent_and_two_primary_evaluations_verified':True,'frozen_preflight':checked})
        for name,ckpt,evaluation in [
            ('parent-012000',BASE/'runs/baseline-seed0/ckpt_012000.pt',BASE/'runs/baseline-evaluation.json'),
            ('selected',checkpoint,EXPERIMENT/'evaluations/016000.json')]:
            monitor.run('diagnostic-'+name,[str(OUT/'diagnose_observed.py'),'--checkpoint',str(ckpt),
                '--evaluation',str(evaluation),'--output',str(OUT/'diagnostics'/name)],deadline,1000)
        observed=read(OUT/'diagnostics/selected/summary.json')
        original=read(BASE/'runs/baseline-evaluation.json')
        os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa'
        for index,clip in enumerate(['dance2_subject4','fallAndGetUp1_subject5','fallAndGetUp2_subject3']):
            row=observed['clips'][clip]
            focus=row['seconds'] if row['failed'] else original['per_clip'][clip]['steps']*.02+5
            monitor.run('video-'+str(index),['-m','unisupermimic.tasks.motion_tracking.g1.supermimic.render',
                '--backend','mjwarp','--ckpt',str(checkpoint),'--clip',clip,'--start',str(max(0,focus-10)),
                '--seconds','15','--global-ghost','--output',str(OUT/'videos'/(clip+'.mp4'))],deadline,420)
        record.update(status='complete',selected_diagnostic_macro=observed['observed_evaluation']['norm_return'],
            selected_diagnostic_failed=observed['observed_evaluation']['n_failed'])
        print('AUXILIARY_POSTPROCESS_COMPLETE '+json.dumps(record),flush=True)
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            resource_release_verified=False)
        write_json(OUT/'evidence/postprocess.json',record)


if __name__=='__main__':main()
