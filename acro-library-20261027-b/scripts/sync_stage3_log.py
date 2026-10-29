"""Update local task state only from observed console output, preserving its limits."""
import json,re
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'evidence/stage3-latest-log.txt';log=source.read_text(encoding='utf8')
    path=ROOT/'execution.json';state=json.loads(path.read_text(encoding='utf8'));phase=state['stage3']
    checkpoints=[int(value) for value in re.findall(r'ckpt_(\d+)\.pt',log)]
    if checkpoints:
        latest=max(checkpoints);assert 26000<=latest<=30000
        phase.update(latest_checkpoint_observed=f'ckpt_{latest:06d}.pt',new_updates_observed=latest-26000)
    for line in log.splitlines():
        if line.startswith('UNIFIED_STAGE3_UPDATE_PROOF '):
            proof=json.loads(line.split(' ',1)[1]);assert proof['new_updates']==4000 and proof['normalizer_tensors_unchanged']
            (ROOT/'evidence/stage3-update-proof-from-console.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
            phase.update(new_updates_observed=4000,training_update_proof='evidence/stage3-update-proof-from-console.json')
            state['stage']='stage3_training_updates_verified_post_evaluations_running'
        if line.startswith('UNIFIED_STAGE3_COMPLETE '):
            result=json.loads(line.split(' ',1)[1]);assert result['status']=='updates_corrected55_and_raw15_evaluation_complete'
            (ROOT/'evidence/stage3-complete-from-console.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
            phase.update(program_status=result['status'],corrected_groups=result['corrected'],raw_new_groups=result['raw_new'],full_checkpoint_sha256=result['full_checkpoint_sha256'])
            state['stage']='stage3_program_complete_full_results_pending_retrieval'
    time=datetime.fromtimestamp(source.stat().st_mtime,timezone(timedelta(hours=8))).isoformat()
    phase['last_console_log_read_at']=time;state['updated_at']=time
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'stage':state['stage'],'new_updates_observed':phase.get('new_updates_observed'),
                      'latest_checkpoint_observed':phase.get('latest_checkpoint_observed'),'last_console_log_read_at':time}),flush=True)

if __name__=='__main__':main()
