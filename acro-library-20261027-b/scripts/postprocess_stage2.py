"""Finite local assessment after verified recovery; never starts cloud resources."""
import argparse,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(path.read_text(encoding='utf8'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-render',action='store_true');args=parser.parse_args()
    verified=read(ROOT/'evidence/local-transfer-verification-stage2-4000.json');assert verified['status']=='archive_and_all_manifest_files_verified'
    checkpoint=ROOT/'policy/unified-h100-stage2-4000.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt=ROOT/'evidence/postprocess-stage2.json';started=time.monotonic();completed=[]
    invocation=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if receipt.exists():
        history=ROOT/'evidence'/f'postprocess-stage2-history-{invocation}.json';assert not history.exists();history.write_bytes(receipt.read_bytes())
    def record(status,**extra):
        value={'status':status,'checkpoint_sha256':sha,'invocation':invocation,'completed_commands':completed,'updated_at':datetime.now(timezone.utc).isoformat(),**extra}
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    record('running')
    def run(script,argv,artifact=None,render=False):
        if artifact and (ROOT/artifact).exists():return
        assert time.monotonic()-started<3600,'Finite local postprocessing budget exhausted'
        command=[sys.executable,'-u',str(ROOT/'scripts'/script),*argv]
        environment=os.environ.copy();environment.update(PYTHONUTF8='1',MUJOCO_GL='glfw' if render else 'disable')
        print('STAGE2_LOCAL_START '+json.dumps({'script':script,'arguments':argv}),flush=True)
        log=ROOT/'evidence'/('stage2-local-'+invocation+'-'+script.removesuffix('.py')+'-'+str(len(completed))+'.log')
        with log.open('w',encoding='utf8') as output:subprocess.run(command,env=environment,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=min(1200,3600-(time.monotonic()-started)))
        if artifact:assert (ROOT/artifact).exists(),artifact
        completed.append({'script':script,'arguments':argv,'log':str(log.relative_to(ROOT))});record('running')
        print('STAGE2_LOCAL_END '+json.dumps({'script':script,'artifact':artifact}),flush=True)
    try:
        cloud_modes=['unified-stage2-corrected-cuda-all55','unified-stage2-raw-cuda-new15']
        expected_full=read(ROOT/'evidence/gpu-stage2-4000.json')['full_checkpoint_sha256']
        for mode in cloud_modes:assert read(ROOT/f'runs/{mode}/summary.json')['checkpoint_sha256']==expected_full
        cpu_mode='unified-stage2-corrected-cpu-all55'
        run('evaluate_ground_variant.py',['--checkpoint',str(checkpoint),'--mode',cpu_mode,'--group','all','--device','cpu'],f'runs/{cpu_mode}/summary.json')
        assert read(ROOT/f'runs/{cpu_mode}/summary.json')['checkpoint_sha256']==sha
        for mode in cloud_modes+[cpu_mode]:
            run('assess_tracking.py',['--run',mode],f'runs/{mode}/tracking-assessment.json')
            run('audit_physical_contacts.py',['--run',mode],f'runs/{mode}/contact-review.json')
            run('combine_acceptance.py',['--run',mode],f'runs/{mode}/combined-assessment.json')
        for script,mode in [('evaluate_sequences.py','stage2-frozen-continuous-v1'),('evaluate_sequences_corrected.py','stage2-corrected-continuous-v1')]:
            run(script,['--checkpoint',str(checkpoint),'--mode',mode,'--device','cpu'],f'runs/{mode}/summary.json')
            assert read(ROOT/f'runs/{mode}/summary.json')['checkpoint_sha256']==sha
            run('assess_sequences.py',['--run',mode],f'runs/{mode}/continuous-assessment.json')
        if not args.skip_render:
            for mode,names in [('unified-stage2-corrected-cuda-all55',[Path(row['path']).stem for row in read(ROOT/'library-manifest.json')['files']]),
                               ('unified-stage2-raw-cuda-new15',['90_14_stageii_unitree_g1','90_15_stageii_unitree_g1']),
                               ('stage2-frozen-continuous-v1',['seed-0','seed-1','seed-2']),('stage2-corrected-continuous-v1',['seed-0','seed-1','seed-2'])]:
                for name in names:
                    run('render_library.py',['--run',mode,'--clips',name],f'videos/{mode}/{name}.json',render=True)
                    video=read(ROOT/f'videos/{mode}/{name}.json');target=ROOT/video['file']
                    assert video['complete_decode_passed'] and target.stat().st_size==video['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==video['sha256']
        corrected=read(ROOT/'runs/unified-stage2-corrected-cuda-all55/combined-assessment.json')
        raw=read(ROOT/'runs/unified-stage2-raw-cuda-new15/combined-assessment.json')
        cpu=read(ROOT/f'runs/{cpu_mode}/summary.json');old=cpu['groups']['lafan'];baseline_summary=read(ROOT/'runs/old-16k-all55-v3/summary.json');baseline=baseline_summary['groups']['lafan']
        regression_limits=read(ROOT/'acceptance-contract.json')['old_lafan_regression']
        assert cpu['device']==baseline_summary['device']=='cpu' and cpu['backend']==baseline_summary['backend']
        regression=40-old['passed']<=regression_limits['hard_failed_clips_max'] and baseline['macro_return_original']-old['macro_return_original']<=regression_limits['macro_return_drop_from_matched_original_16k_max']
        old_cuda=read(ROOT/'runs/unified-stage2-corrected-cuda-all55/summary.json')['groups']['lafan']
        old_cuda_gate=old_cuda['passed']==40
        raw_sequence=read(ROOT/'runs/stage2-frozen-continuous-v1/continuous-assessment.json')
        corrected_sequence=read(ROOT/'runs/stage2-corrected-continuous-v1/continuous-assessment.json')
        result={'corrected_new_combined_passed':corrected['combined_engineering_passed'],'raw_new_combined_passed':raw['combined_engineering_passed'],
                'new_denominator':15,'old_CPU_regression_passed':regression,'old_CPU_groups':cpu['groups'],
                'old_CUDA_original_hard_gate_passed_all40':old_cuda_gate,'old_CUDA_group':old_cuda,
                'raw_continuous_engineering_passed':raw_sequence['continuous_engineering_passed'],'corrected_continuous_engineering_passed':corrected_sequence['continuous_engineering_passed'],
                'continuous_denominator':3,'all_failed_tails_retained':True,'raw_reference_goal_complete':raw['combined_engineering_passed']==15 and regression and old_cuda_gate and raw_sequence['continuous_engineering_passed']==3,
                'corrected_variant_goal_complete':corrected['combined_engineering_passed']==15 and regression and old_cuda_gate and corrected_sequence['continuous_engineering_passed']==3,
                'paper_equivalent_reproduction_claimed':False,'cloud_resource_release_verified_separately':True,'video_generation_pending':args.skip_render}
        state=read(ROOT/'execution.json');state['next_stage']['physical_assessment']=result
        state['stage']='stage2_physical_assessments_complete_rendering_pending' if args.skip_render else 'stage2_all_local_assessments_complete'
        state['full_goal_complete']=result['raw_reference_goal_complete'] and not args.skip_render
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        (ROOT/'execution.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        run('build_report.py',[])
        record('physical_assessments_complete_rendering_pending' if args.skip_render else 'all_requested_local_assessments_complete',results=result)
        print('STAGE2_LOCAL_COMPLETE '+json.dumps(result),flush=True)
    except BaseException as error:record('failed_or_interrupted',error=repr(error));raise

if __name__=='__main__':main()
