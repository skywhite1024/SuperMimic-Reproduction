"""One Q follow-up clip using unchanged I evaluate_clip; no precision override."""
from pathlib import Path
import argparse,json,signal,sys,time,traceback

def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model',choices=['parent','f0','f1'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();deadline=started+a.max_seconds;plan=json.loads(a.plan.read_text())
    def finite_stop(signum,frame):raise TimeoutError('Q finite process deadline signal')
    signal.signal(signal.SIGTERM,finite_stop)
    a.output.mkdir(parents=True,exist_ok=False)
    report={'status':'starting','model':a.model,'selection':'J failure triggered follow-up; not random held-out',
            'full_protocol_evaluation_complete':False,'formal_experiment':False,'sports_quality_acceptance':False,
            'device':'cuda','num_envs':1,'action_mode':'raw','seed':0,'post_initialization_resets':0,
            'precision_override_requested':False,'max_seconds':a.max_seconds,'clips':[]}
    try:
        sys.path.insert(0,str(a.runtime))
        import torch
        from evaluate import evaluate_clip,prepare_checkpoint,aggregate,write_json
        from sports_runtime import load_manifest,sha
        torch.set_num_threads(4)
        assert torch.cuda.device_count()==1,'Each Q process must see exactly one GPU'
        name=torch.cuda.get_device_name(0);assert 'H100' in name,name
        m=next(x for x in plan['models'] if x['id']==a.model)
        assert sha(m['path'])==m['sha256'],'Frozen I checkpoint mismatch'
        _,sources,_=load_manifest(a.manifest)
        assert len(sources)==1 and sources[0]['path'].stem==plan['target_clip']
        saved,prep=prepare_checkpoint(torch.load(m['path'],map_location='cpu',weights_only=True),m['feedback'])
        report.update(checkpoint_sha256=m['sha256'],checkpoint_preparation=prep,gpu=name,
                      runtime_code_sha256={n:sha(a.runtime/n) for n in ['evaluate.py','sports_runtime.py']})
        write_json(a.output/'summary.json',report)
        row=evaluate_clip(sources[0],saved,device='cuda',feedback=m['feedback'],seed=0,
                          action_mode='raw',output=a.output/'clip-0000-fallAndGetUp1_subject5',
                          max_steps=None,deadline=deadline)
        report.update(clips=[row],aggregate=aggregate([row]),status='diagnostic_finished',
                      native_outcome_observed=row['status'] in ['completed','native_hard_failure'],
                      full_reference_completed_without_failure=bool(row['completed_without_native_failure']))
    except Exception as exc:
        report.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc())
    finally:
        report['wall_seconds']=time.perf_counter()-started
        tmp=a.output/'summary.json.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        tmp.replace(a.output/'summary.json')
        print('Q_SINGLE_RECEIPT '+json.dumps({k:report.get(k) for k in ['model','status','native_outcome_observed','aggregate']}),flush=True)
    return 0 if report.get('native_outcome_observed') else 2

if __name__=='__main__':raise SystemExit(main())
