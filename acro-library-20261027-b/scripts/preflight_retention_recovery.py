"""CPU-only dependency and complete-input verification before any resumed GPU work."""
import hashlib,importlib.metadata,json,os,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    assert ROOT.resolve().is_relative_to(Path('/mnt/afs/USER_ID'))
    assert Path(sys.prefix).resolve()==Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024/uni/.venv').resolve()
    plan_path=ROOT/'retention-recovery-1030-plan.json';plan=json.loads(plan_path.read_text())
    for name,digest in plan['files'].items():assert sha(ROOT/name)==digest,name
    frozen=json.loads((ROOT/'retention-trial-v1-plan.json').read_text())
    for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest,name
    tracked=['torch','numpy','mujoco','mujoco-warp','warp-lang','unilab','hydra-core']
    before={name:importlib.metadata.version(name) for name in tracked}
    assert sys.version_info[:2]==(3,12) and platform.machine()=='x86_64'
    wheel=ROOT/plan['scipy_wheel']['path'];assert sha(wheel)==plan['scipy_wheel']['sha256']
    try:scipy_before=importlib.metadata.version('scipy')
    except importlib.metadata.PackageNotFoundError:scipy_before=None
    if scipy_before is None:
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index',str(wheel)],check=True,timeout=180)
    else:assert scipy_before=='1.17.0'
    import numpy as np
    from scipy.spatial.transform import Rotation
    assert importlib.metadata.version('scipy')=='1.17.0' and np.array_equal(Rotation.from_quat([0,0,0,1]).as_matrix(),np.eye(3))
    after={name:importlib.metadata.version(name) for name in tracked};assert before==after
    os.environ.update(MUJOCO_GL='disable',HF_HUB_OFFLINE='1',WANDB_MODE='offline',SUPERMIMIC_UNI=str(ROOT.parent/'lafan1-20261024/uni'))
    for mode in ['continuous-frozen-v1','continuous-corrected-v1']:
        index=json.loads((ROOT/mode/'index.json').read_text());assert index['seeds']==[0,1,2]
        for row in index['sequences']:
            file=ROOT/mode/f'seed-{row["seed"]}'/'sequence.csv';assert sha(file)==row['reference_sha256']
            matrix=np.loadtxt(file,delimiter=',');assert matrix.shape[0]>=6001 and matrix.shape[1]==36 and np.isfinite(matrix).all()
    from evaluate_sequences import sequence_env
    lengths={}
    for mode in ['continuous-frozen-v1','continuous-corrected-v1']:
        environment=sequence_env([ROOT/mode/f'seed-{seed}'/'sequence.csv' for seed in [0,1,2]],101,'cpu')
        try:
            value=environment.supermimic_mdp.motion.clip_len.tolist();assert len(value)==3 and min(value)>=6001
            lengths[mode]=value
        finally:environment.close();environment.backend.close()
    parent='retention-v1-parent-bound20-corrected-cuda-all55'
    summary=json.loads((ROOT/'runs'/parent/'summary.json').read_text())
    assert summary['checkpoint_sha256']==frozen['initial_full_sha256'] and len(summary['clips'])==55
    assert summary['post_initialization_resets']==0 and summary['controller_variant']=='action-output-bound-20-v1'
    for script,output in [('assess_tracking.py','tracking-assessment.json'),('audit_physical_contacts.py','contact-review.json'),('combine_acceptance.py','combined-assessment.json')]:
        if not (ROOT/'runs'/parent/output).exists():
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),'--run',parent],check=True,timeout=180)
    result={'status':'all_dependencies_inputs_sequence_constructors_and_parent_contact_audit_passed','checked_at':datetime.now(timezone.utc).isoformat(),
            'recovery_plan_sha256':sha(plan_path),'original_trial_plan_sha256':sha(ROOT/'retention-trial-v1-plan.json'),
            'packages_before':before,'packages_after':after,'scipy_before':scipy_before,'scipy_after':'1.17.0',
            'six_exact_fixed_sequences_verified':True,'sequence_constructor_lengths':lengths,'all_reference_bytes_unchanged':True,
            'parent55_trace_sha256':sha(ROOT/'runs'/parent/'trajectory.npz'),'parent55_summary_sha256':sha(ROOT/'runs'/parent/'summary.json'),
            'parent_contact_and_tracking_completed_on_CPU_without_resimulation':True,'training_updates_performed':0}
    target=ROOT/'evidence/retention-1030-dependency-preflight.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    print('RETENTION_DEPENDENCY_PREFLIGHT_PASS '+json.dumps(result),flush=True)

if __name__=='__main__':main()
