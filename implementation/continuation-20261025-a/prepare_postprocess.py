"""Preserve the failed strict replay check and prepare an independent auxiliary run."""
from pathlib import Path
import base64
import gzip
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent
REC=ROOT/'postprocess-20261026-a'
REC.mkdir(exist_ok=True)


def replace_once(text, before, after):
    assert text.count(before)==1, before[:100]
    return text.replace(before,after)


diagnostic=(ROOT/'diagnose.py').read_text(encoding='utf-8')
diagnostic=replace_once(diagnostic,'from audit_common import file_hash, write_json',
    'import sys\nsys.path.insert(0,str(Path(__file__).resolve().parent.parent))\nfrom audit_common import file_hash, write_json')
before='''        for index, name in enumerate(motion.names):
            row = expected['per_clip'][name]
            assert int(steps[index]) == row['steps'] and bool(failed[index]) == row['failed'], name
            assert abs(float(normalized[index]) - row['norm_return']) < 1e-5, name
'''
after='''        actual = {'norm_return': float(normalized.mean()),
                  'norm_return_framew': float(returns.sum()/((motion.clip_len-1).float()*.1).sum()),
                  'n_failed': int(failed.sum()),
                  'failed': [name for index,name in enumerate(motion.names) if bool(failed[index])],
                  'per_clip': {name: {'norm_return': float(normalized[index]), 'steps': int(steps[index]),
                      'failed': bool(failed[index]), 'full_clip_steps': int(motion.clip_len[index])-1}
                      for index,name in enumerate(motion.names)}}
        comparison=[]
        for index, name in enumerate(motion.names):
            row = expected['per_clip'][name]
            new = actual['per_clip'][name]
            assert 0<new['steps']<=new['full_clip_steps']
            if not new['failed']: assert new['steps']==new['full_clip_steps']
            comparison.append({'clip': name, 'return_delta': new['norm_return']-row['norm_return'],
                'steps_delta': new['steps']-row['steps'], 'failed_same': new['failed']==row['failed'],
                'strict_match_1e_minus5': new['steps']==row['steps'] and new['failed']==row['failed'] and abs(new['norm_return']-row['norm_return'])<1e-5})
        strict_match=all(row['strict_match_1e_minus5'] for row in comparison)
        write_json(args.output/'observed-evaluation.json',actual)
        write_json(args.output/'primary-comparison.json',{'reference_evaluation': str(args.evaluation),
            'checkpoint_sha256': file_hash(args.checkpoint), 'strict_all40_match': strict_match,
            'macro_delta': actual['norm_return']-expected['norm_return'], 'per_clip': comparison,
            'boundary': 'Auxiliary independent numerical rollout; no tolerance or acceptance metric was relaxed. Failed strict replay checks remain false.'})
'''
diagnostic=replace_once(diagnostic,before,after)
diagnostic=replace_once(diagnostic,"'failed': expected['per_clip'][name]['failed']", "'failed': actual['per_clip'][name]['failed']")
diagnostic=replace_once(diagnostic,"'same_all40_rollout_metrics_reproduced': True, 'clips': summaries,",
    "'same_all40_rollout_metrics_reproduced': strict_match, 'clips': summaries,\n            'independent_trace_contract_checked': True, 'observed_evaluation': actual, 'primary_comparison': comparison,")
diagnostic=replace_once(diagnostic,"'boundary': 'Joint-position tracking error and action magnitude/rate are distinct;diagnostics do not alter metrics or termination.'",
    "'boundary': 'Independent numerical rollout with frozen policy,MDP and thresholds;not asserted identical to primary results. Exact gate-versus-termination identity checked. Joint tracking error and action magnitude/rate are distinct.'")
(REC/'diagnose_observed.py').write_text(diagnostic,encoding='utf-8',newline='\n')

collector=(ROOT/'collect_continuation.py').read_text(encoding='utf-8')
collector=replace_once(collector,'from audit_common import file_hash, write_json',
    'import sys\nsys.path.insert(0,str(Path(__file__).resolve().parent.parent))\nfrom audit_common import file_hash, write_json')
collector=replace_once(collector,'root = Path(__file__).resolve().parent','root = Path(__file__).resolve().parent.parent')
collector=replace_once(collector,"    for pattern in ['args.json'",'''    auxiliary=root/'postprocess-20261026-a'
    auxiliary_state=json.loads((auxiliary/'evidence/postprocess.json').read_text())
    assert auxiliary_state['status']=='complete', 'Auxiliary outputs must be terminal and complete'
    for path in auxiliary.rglob('*'):
        if path.is_file() and path.suffix in ['.py','.json','.jsonl','.txt','.gz','.mp4']: add(path)
    for pattern in ['args.json' '''.rstrip())
collector=replace_once(collector,"'target_confirmed':experiment.get('target_confirmed',False),'original_iteration':12000,",
    "'target_confirmed':experiment.get('target_confirmed',False),'original_iteration':12000,\n             'auxiliary_status':auxiliary_state['status'], 'primary_pipeline_status_preserved':experiment['status'],\n             'strict_replay_failure_preserved':True, 'diagnostic_rollouts_are_independent':True,")
collector=replace_once(collector,"'final_iteration':experiment.get('final_iteration')", "'final_iteration':json.loads((root/'evidence/continuation-proof.json').read_text())['after']['iteration']")
(REC/'collect_complete.py').write_text(collector,encoding='utf-8',newline='\n')

files={name:hashlib.sha256((REC/name).read_bytes()).hexdigest() for name in ['diagnose_observed.py','run_postprocess.py','collect_complete.py']}
payload=base64.b64encode(gzip.compress(json.dumps({name:(REC/name).read_text(encoding='utf-8') for name in files}).encode(),mtime=0)).decode()
cloud='/mnt/afs/USER_ID/SuperMimic/lafan1-20261024/experiments/continuation-20261025-a/postprocess-20261026-a'
command='''set -e
export SUPERMIMIC_BASE=/mnt/afs/USER_ID/SuperMimic/lafan1-20261024
export MUJOCO_GL=disable WANDB_MODE=offline HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
unset PYOPENGL_PLATFORM
cd /mnt/afs/USER_ID/SuperMimic/lafan1-20261024/uni
python3 - <<'PY'
import pathlib,base64,gzip,hashlib,json
root=pathlib.Path(CLOUD)
assert not root.exists(),'Refusing duplicate auxiliary run'
root.mkdir()
expected=EXPECTED
payload=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
for name,source in payload.items():
 data=source.encode();assert hashlib.sha256(data).hexdigest()==expected[name],name
 (root/name).write_bytes(data)
print('AUXILIARY_SOURCE_DEPLOYED_NO_TRAINING',flush=True)
PY
export DEBIAN_FRONTEND=noninteractive
timeout --signal=TERM --kill-after=10s 300s bash -c 'set -euo pipefail; apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 update; apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 install -y --no-install-recommends libosmesa6=25.1.7-1ubuntu2~24.04.2'
timeout --signal=TERM --kill-after=30s 3450s .venv/bin/python -u CLOUD/run_postprocess.py
'''.replace('CLOUD',repr(cloud),1).replace('EXPECTED',repr(files),1).replace('PAYLOAD',repr(payload),1).replace('CLOUD',cloud)
(REC/'startup.command.txt').write_text(command,encoding='utf-8',newline='\n')
subprocess.run([r'[local]/Program Files\Git\bin\bash.exe','-n',str(REC/'startup.command.txt')],check=True)
for name in files: compile((REC/name).read_text(),name,'exec')
record={'files':files,'command_bytes':len(command.encode()),'command_sha256':hashlib.sha256(command.encode()).hexdigest(),
        'training_executed':False,'purpose':'Auxiliary independent rollout diagnostics and videos; preserve failed strict match as false',
        'maximum_program_seconds_including_setup_and_kill_grace':3790,'gpus':1,'replicas':1,'max_restarts':0,'node_ssh':False}
(REC/'preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
