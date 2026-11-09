"""Fresh Q orchestration only: parent on GPU0, F0 then F1 on GPU1."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,base64,hashlib,json,os,shutil,signal,subprocess,sys,threading,time

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)
def emit(kind,obj):
    raw=json.dumps(obj,separators=(',',':'),allow_nan=False).encode();encoded=base64.b64encode(raw).decode()
    parts=[encoded[i:i+280] for i in range(0,len(encoded),280)]
    for i,b in enumerate(parts):
        line='Q_JSON '+json.dumps({'k':kind,'i':i,'n':len(parts),'sha':hashlib.sha256(raw).hexdigest(),'b':b},separators=(',',':'))
        assert len(line)<500;print(line,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    start=time.monotonic();deadline=start+860;plan=read(a.plan);work=Path(plan['output']);work.mkdir(parents=True,exist_ok=False)
    children={};lock=threading.Lock();stop=threading.Event()
    def on_term(signum,frame):
        stop.set()
        with lock:
            for child in children.values():
                if child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,on_term)
    report={'status':'preflight','selection':plan['selection'],'J_trigger':plan['J_trigger'],
            'fixed_denominator':3,'target_clip':plan['target_clip'],'training_performed':False,
            'formal_experiment':False,'independent_E_star':False,'precision_override_requested':False,
            'all_three_native_outcomes_observed':False,'models':{n:{'status':'not_run'} for n in ['parent','f0','f1']}}
    try:
        here=Path(__file__).resolve().parent
        for name,digest in plan['uploaded_code_sha256'].items():assert sha(here/name)==digest,'Q code SHA mismatch: '+name
        runtime=work/'runtime';runtime.mkdir()
        for name,digest in plan['I_code_sha256'].items():
            src=Path(plan['I_runtime'])/name;assert sha(src)==digest,'I code SHA mismatch: '+name
            if name in ['evaluate.py','sports_runtime.py']:shutil.copy2(src,runtime/name)
        source_spec=read(plan['I_source_spec'])
        current={x['id']:x for x in source_spec['models']}
        for expected in plan['models']:
            actual=current[expected['id']]
            for key in ['path','feedback','sha256']:assert actual[key]==expected[key],'I spec identity changed'
            assert actual.get('available') and sha(actual['path'])==expected['sha256'],'Frozen checkpoint SHA mismatch'
        lockfile=read(Path(plan['I_runtime'])/'runtime-source-lock.json')
        for relative,digest in lockfile.items():assert sha(Path(plan['cached_runtime'])/relative)==digest,'Cached runtime changed: '+relative
        target=plan['target_source'];assert sha(target['path'])==target['sha256'],'Target CSV changed'
        manifest=work/'single-clip-manifest.json'
        write(manifest,{'reference_variant':'I_frozen_old_reference_Q_J_failure_followup','clips':[target],
                        'selection':plan['selection'],'independent_E_star':False})
        shutil.copy2(a.plan,work/'q-plan.json')
        report.update(status='running',preflight={'code_sha_verified':True,'all_three_frozen_checkpoints_verified':True,
            'target_reference_verified':True,'cached_code_lock_items':len(lockfile),'I_spec_sha256':sha(plan['I_source_spec'])})
        write(work/'q-receipt.json',report);emit('preflight',report['preflight'])
        def run_one(name,gpu):
            budget=min(600.,deadline-time.monotonic()-15.)
            if budget<=0 or stop.is_set():
                result={'status':'not_run_total_deadline','gpu_slot':gpu}
            else:
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                     'SUPERMIMIC_UNI':plan['uni'],'WANDB_MODE':'offline','HF_HUB_OFFLINE':'1','MUJOCO_GL':'disable'}
                command=[sys.executable,'-u',str(here/'worker_q.py'),'--plan',str(a.plan.resolve()),'--runtime',str(runtime),
                         '--manifest',str(manifest),'--model',name,'--output',str(work/name),'--max-seconds',str(budget)]
                with (work/(name+'.log')).open('w') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                    with lock:children[name]=child
                    try:code=child.wait(timeout=budget+8)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:code=child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=2)
                    with lock:children.pop(name,None)
                summary=work/name/'summary.json'
                result={'status':'process_finished','process_exit_code':code,'gpu_slot':gpu,'soft_seconds':budget,
                        'summary':read(summary) if summary.is_file() else None,'summary_sha256':sha(summary) if summary.is_file() else None}
            with lock:
                report['models'][name]=result;write(work/'q-receipt.json',report);emit(name,result)
            return result
        def lane1():
            run_one('f0',1);run_one('f1',1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks=[pool.submit(run_one,'parent',0),pool.submit(lane1)]
            for task in tasks:task.result()
        report['all_three_native_outcomes_observed']=all(bool(r.get('summary') and r['summary'].get('native_outcome_observed')) for r in report['models'].values())
        report['status']='followup_diagnostic_finished' if report['all_three_native_outcomes_observed'] else 'incomplete_followup_diagnostic'
    except Exception as exc:
        report.update(status='failed_or_interrupted',error=repr(exc));on_term(None,None)
    finally:
        report['wall_seconds']=time.monotonic()-start
        write(work/'q-receipt.json',report)
        emit('final',{k:v for k,v in report.items() if k!='models'})
    return 0 if report['all_three_native_outcomes_observed'] else 2

if __name__=='__main__':raise SystemExit(main())
