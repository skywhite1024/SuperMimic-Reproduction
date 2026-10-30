"""Package a finite operational recovery; preserve the frozen training implementation."""
import ast
import base64
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    trial = ROOT / 'retention-trial-v1-plan.json'
    assert sha(trial) == 'e737aacaf113cf602be0eb6cb7b64a5b47e14451acdb003a4c19d9cbaa575a53'
    frozen = json.loads(trial.read_text(encoding='utf8'))
    for name, expected in frozen['files'].items():
        assert sha(ROOT / name) == expected, name
    dependency = json.loads((ROOT / 'evidence/retention-1030-dependency-preflight.json').read_text())
    assert dependency['status'] == 'all_dependencies_inputs_sequence_constructors_and_parent_contact_audit_passed'
    assert dependency['training_updates_performed'] == 0
    previous = json.loads((ROOT / 'evidence/gpu-retention-v1-1000.json').read_text())
    assert previous['status'] == 'failed_or_interrupted' and 'audit_physical_contacts.py' in previous['error']
    assert 0 < previous['wall_seconds'] < 1200
    assert not (ROOT / 'runs/unified-retention-v1-h100-1000').exists()
    parent = 'runs/retention-v1-parent-bound20-corrected-cuda-all55/'
    reuse = [parent + name for name in ['summary.json', 'trajectory.npz', 'tracking-assessment.json',
                                        'contact-review.json', 'contact-samples.json', 'combined-assessment.json']]
    operational_files = ['retention-trial-v1-plan.json', 'retention-recovery-1030-plan.json',
                         'evidence/retention-1030-dependency-preflight.json', 'evidence/gpu-retention-v1-1000.json',
                         'scripts/run_retention_trial.py', 'scripts/audit_sequence_contacts.py',
                         'runs/engineering-retention-v2-cpu-6/proof.json'] + reuse
    original = (ROOT / 'scripts/run_retention_trial.py').read_text(encoding='utf8')
    preflight = '''
def operational_preflight(plan):
    import importlib.metadata
    operation=json.loads((ROOT/'retention-resume-1030b-plan.json').read_text())
    for name,digest in operation['files'].items():assert sha(ROOT/name)==digest,name
    dependency=json.loads((ROOT/'evidence/retention-1030-dependency-preflight.json').read_text())
    assert dependency['status']=='all_dependencies_inputs_sequence_constructors_and_parent_contact_audit_passed'
    assert dependency['training_updates_performed']==0 and dependency['six_exact_fixed_sequences_verified']
    assert dependency['packages_before']==dependency['packages_after']
    for name,version in dependency['packages_after'].items():assert importlib.metadata.version(name)==version,name
    assert importlib.metadata.version('scipy')=='1.17.0'
    from scipy.spatial.transform import Rotation
    assert Rotation.from_quat([0,0,0,1]).as_matrix().shape==(3,3)
    recovery=json.loads((ROOT/'retention-recovery-1030-plan.json').read_text())
    assert sha(ROOT/'retention-recovery-1030-plan.json')==dependency['recovery_plan_sha256']
    for name,digest in recovery['files'].items():assert sha(ROOT/name)==digest,name
    previous=json.loads((ROOT/'evidence/gpu-retention-v1-1000.json').read_text())
    assert previous['status']=='failed_or_interrupted' and 'audit_physical_contacts.py' in previous['error']
    assert previous['wall_seconds']==operation['previous_program_wall_seconds']
    assert operation['maximum_new_updates']==1000 and operation['new_program_seconds_max']==4200
    assert operation['previous_program_wall_seconds']+operation['new_program_seconds_max']<=5400
    parent=json.loads((ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/summary.json').read_text())
    assert parent['checkpoint_sha256']==plan['initial_full_sha256']
    assert parent['device']=='cuda' and parent['observation_dim']==101 and parent['one_policy']
    assert len(parent['clips'])==55 and parent['groups']['new']['denominator']==15 and parent['groups']['lafan']['denominator']==40
    assert parent['controller_variant']=='action-output-bound-20-v1' and parent['reference_variant']=='ground-calibration-v1'
    assert not parent['auto_reset'] and parent['post_initialization_resets']==0 and parent['no_controlled_robot_state_writes_after_t0']
    combined=json.loads((ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/combined-assessment.json').read_text())
    assert combined['new_denominator']==15 and combined['combined_engineering_passed']==10
    assert sha(ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/summary.json')==dependency['parent55_summary_sha256']
    assert sha(ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/trajectory.npz')==dependency['parent55_trace_sha256']
    assert not (ROOT/'runs/unified-retention-v1-h100-1000').exists(),'Never duplicate any candidate updates'
    for mode in ['retention-v1-parent-bound20-raw-cuda-new15','retention-v1-parent-bound20-raw-continuous-cuda',
                 'retention-v1-parent-bound20-corrected-continuous-cuda','retention-v1-corrected-cuda-all55',
                 'retention-v1-raw-cuda-new15','retention-v1-raw-continuous-cuda','retention-v1-corrected-continuous-cuda']:
        assert not (ROOT/'runs'/mode).exists(),f'Inspect any unknown outcome before retry: {mode}'
    return operation

'''
    text = original.replace('def main():', preflight + 'def main():', 1)
    anchor = "    assert plan['status']=='frozen_before_v2_engineering_and_cuda_candidate_assessment'"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\n    operation=operational_preflight(plan)', 1)
    text = text.replace("receipt_path=ROOT/'evidence/gpu-retention-v1-1000.json'",
                        "receipt_path=ROOT/'evidence/gpu-retention-v1-1000-resume1030b.json'", 1)
    text = text.replace("'maximum_seconds':5400", "'maximum_seconds':4200", 1)
    text = text.replace("'full_goal_complete':False}",
                        "'full_goal_complete':False,'operational_resume_plan_sha256':sha(ROOT/'retention-resume-1030b-plan.json'),'previous_program_wall_seconds':operation['previous_program_wall_seconds'],'completed_parent55_reused_without_resimulation':True}", 1)
    text = text.replace("remaining=5200-(time.monotonic()-started)",
                        "remaining=min(4200,5400-operation['previous_program_wall_seconds'])-60-(time.monotonic()-started)", 1)
    anchor = "        evaluate(initial,'retention-v1-parent-bound20-corrected-cuda-all55','corrected',parent=True)"
    assert text.count(anchor) == 1
    text = text.replace(anchor,
                        "        print('RETENTION_COMPLETED_PARENT55_REUSED '+json.dumps({'files':operation['reused_parent_artifacts'],'source_failed_attempt':'d6a6cfc61c094fac88d66b88fabf554e'}),flush=True)", 1)
    text = text.replace("policy_sha256=sha(policy),matched_parent_groups=parent['groups']",
                        "policy_sha256=sha(policy),combined_attempt_program_wall_seconds=operation['previous_program_wall_seconds']+time.monotonic()-started,matched_parent_groups=parent['groups']", 1)
    ast.parse(text)
    driver = ROOT / 'scripts/run_retention_resume1030b.py'
    assert not driver.exists()
    driver.write_text(text, encoding='utf8', newline='\n')
    operational_files.append('scripts/run_retention_resume1030b.py')
    operation = {'status': 'frozen_before_operational_resume1030b', 'frozen_at': datetime.now(timezone.utc).isoformat(),
                 'original_trial_plan_sha256': sha(trial), 'previous_program_wall_seconds': previous['wall_seconds'],
                 'maximum_new_updates': 1000, 'wm_only_updates': 200, 'new_program_seconds_max': 4200,
                 'combined_attempt_program_seconds_max': 5400, 'reused_parent_artifacts': reuse,
                 'training_or_reference_or_physics_changes': False, 'no_automatic_second_training_block': True,
                 'files': {name: sha(ROOT / name) for name in operational_files}}
    operation_path = ROOT / 'retention-resume-1030b-plan.json'
    assert not operation_path.exists()
    operation_path.write_text(json.dumps(operation, indent=2) + '\n', encoding='utf8')
    # Reuse the original read-only pinned-source preamble; only its operational payload and entry point change.
    startup = (ROOT / 'deployment/gpu-retention-v1-1028.command.txt').read_text(encoding='utf8')
    preamble = startup[startup.index("base=pathlib.Path('/mnt/afs/"):startup.index("print('RETENTION_PINNED_SOURCE_")]
    payload_names = ['retention-resume-1030b-plan.json', 'scripts/run_retention_resume1030b.py']
    payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in payload_names}
    manifest = {name: sha(ROOT / name) for name in payload_names}
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    command = """set -euo pipefail
umask 077
timeout --signal=TERM --kill-after=30s 4200s python3 -u - <<'PY'
import base64,gzip,hashlib,json,os,pathlib
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b')
assert root.resolve().is_relative_to(pathlib.Path('/mnt/afs/USER_ID'))
payload=json.loads(gzip.decompress(base64.b64decode('PAYLOAD')))
manifest=json.loads('MANIFEST')
for name,encoded in payload.items():
    target=root/name;assert target.resolve().is_relative_to(root)
    raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==manifest[name]
    assert not target.exists() or target.read_bytes()==raw,name
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
plan=json.loads((root/'retention-trial-v1-plan.json').read_text())
for name,expected in plan['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
PREAMBLE
print('RETENTION_RESUME_PINNED_SOURCE_REFERENCES_TEACHERS_VERIFIED',flush=True)
os.environ.update(SUPERMIMIC_UNI=str(base/'uni'),MUJOCO_GL='disable',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',WANDB_MODE='offline',PYTHONPATH=str(root/'scripts')+':'+str(base/'uni/src'))
python=str(base/'uni/.venv/bin/python');os.execv(python,[python,'-u',str(root/'scripts/run_retention_resume1030b.py')])
PY
""".replace('PAYLOAD', encoded).replace('MANIFEST', json.dumps(manifest)).replace('PREAMBLE', preamble)
    ast.parse(command.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    target = ROOT / 'deployment/gpu-retention-resume1030b.command.txt'
    assert not target.exists()
    target.write_text(command, encoding='utf8', newline='\n')
    receipt = {'status': 'prepared_not_submitted', 'command': str(target.relative_to(ROOT)),
               'command_sha256': sha(target), 'command_bytes': target.stat().st_size,
               'operational_plan_sha256': sha(operation_path), 'original_trial_plan_sha256': sha(trial),
               'maximum_seconds': 4200, 'maximum_new_updates': 1000, 'gpu_count': 1,
               'original_training_sources_unchanged': True, 'parent55_reused': True,
               'payload_files': manifest, 'fresh_resource_and_network_check_still_required': True}
    (ROOT / 'deployment/gpu-retention-resume1030b-command.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
