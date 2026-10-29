"""Finalize verified local artifacts; keep the next cloud trial explicitly unsubmitted."""
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    completion=read('evidence/action-bound-completion.json')
    assert completion['status']=='all_local_comparisons_and_videos_verified',completion['status']
    assert completion['training_updates']==completion['cloud_resources_started']==0
    comparison=read('evidence/action-bound-matched-comparison.json');assert comparison['status']=='all_four_matched_comparisons_complete'
    for file in ['action-bound-consistency-v1-extension-plan.json','retention-trial-v1-plan.json']:
        for name,digest in read(file)['files'].items():assert sha(ROOT/name)==digest,name
    modes=['stage3-actionbound20-corrected-cpu-all55','stage3-actionbound20-raw-cpu-new15','stage3-actionbound20-frozen-continuous-v1','stage3-actionbound20-corrected-continuous-v1']
    expected_new={Path(row['path']).stem for row in read('library-manifest.json')['files']};assert len(expected_new)==15
    videos=[];traces={}
    for mode in modes:
        summary=read(f'runs/{mode}/summary.json')
        assert summary['checkpoint_sha256']=='4aa7a15f8475d54064db33b3333b772e4976f3ac133f98c03a7e39e49604e3b3'
        assert summary['same_checkpoint_weights_unchanged'] and summary['post_initialization_resets']==0
        if 'clips' in summary:
            assert {row['clip'] for row in summary['clips'] if row['group']=='new'}==expected_new
            expected={row['clip'] for row in summary['clips'] if row['group']=='new' or not row['passed_original_hard_gate']}
            for group,aggregate in summary['groups'].items():
                rows=[row for row in summary['clips'] if row['group']==group]
                assert len(rows)==aggregate['denominator'] and sum(row['passed_original_hard_gate'] for row in rows)==aggregate['passed']
        else:
            assert len(summary['trials'])==3 and all(row['elapsed_seconds']==120 for row in summary['trials'])
            expected={'seed-0','seed-1','seed-2'}
        paths=sorted((ROOT/'videos'/mode).glob('*.json'));assert {path.stem for path in paths}==expected
        traces[mode]={'summary_sha256':sha(ROOT/'runs'/mode/'summary.json'),'trajectory_sha256':sha(ROOT/'runs'/mode/'trajectory.npz')}
        for path in paths:
            video=json.loads(path.read_text());target=ROOT/video['file']
            assert video['complete_decode_passed'] and video['source_run']==mode
            assert target.stat().st_size==video['bytes'] and sha(target)==video['sha256']
            assert (ROOT/video['contact_sheet']).exists()
            videos.append({'receipt':str(path.relative_to(ROOT)),'file':video['file'],'sha256':video['sha256'],'frames':video['frames']})
    assert len(videos)==39
    verified={'status':'all39_ablation_videos_and_all_fixed_comparisons_verified','updated_at':datetime.now(timezone.utc).isoformat(),
              'video_count':39,'video_files':videos,'bound_source_evidence':traces,'checkpoint_weights_unchanged':True,
              'all_original_denominators_and_thresholds_preserved':True,'formal_updates_in_ablation':0,'cloud_resources_started':0,
              'current_cloud_eligibility':'pending_current_location_network_confirmation_and_resource_preflight'}
    (ROOT/'evidence/action-bound-final-verification.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf8')
    state=read('execution.json');state['action_bound_ablation'].update(status='all_numeric_video_and_hash_checks_complete',video_count=39,
        verification='evidence/action-bound-final-verification.json',continuous_failure_figure='figures/continuous-first-failure-diagnostic.png')
    state['stage']='local_ablation_complete_next_cloud_trial_requires_current_location_network_and_resource_check'
    state['full_goal_complete']=False;state['updated_at']=datetime.now(timezone.utc).isoformat()
    (ROOT/'execution.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for script in ['build_action_bound_report.py','build_report.py']:
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts'/script)],check=True,timeout=60)
    for script in ['action-bound-report-script.js','report-script-syntax-check.js']:
        subprocess.run(['node','--check',str(ROOT/'evidence'/script)],check=True,timeout=30)
    class LocalLinks(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attributes):
            for key,value in attributes:
                if key in ['href','src'] and value and not value.startswith(('https:','http:','data:','#')):
                    self.paths.append(value.split('#',1)[0].split('?',1)[0])
    reports={}
    for name in ['action-bound-ablation-report.html','unified-library-report.html']:
        parser=LocalLinks();parser.feed((ROOT/name).read_text(encoding='utf8'))
        for path in parser.paths:assert (ROOT/path).exists(),(name,path)
        reports[name]={'sha256':sha(ROOT/name),'static_local_links_validated':len(parser.paths)}
    qa={'status':'artifact_data_hashes_report_scripts_and_static_links_verified','reports':reports,
        'browser_interaction_verified':False,'browser_reason':'No browser-interaction claim; prior local-file navigation was blocked and no workaround was used.',
        'stage3_video_count':41,'ablation_video_count':39,'goal_complete':False,'next_cloud_trial_submitted':False}
    (ROOT/'evidence/local-report-final-verification.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf8')
    print('LOCAL_ACTION_BOUND_FINALIZED '+json.dumps(qa),flush=True)

if __name__=='__main__':main()
