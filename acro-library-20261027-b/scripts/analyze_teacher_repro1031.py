"""Analyze all three predeclared repeats, including every failed contact gate."""
import hashlib, itertools, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def arrays(mode):
    with np.load(ROOT/'runs'/mode/'trajectory.npz') as z: return {k:z[k] for k in ['state','reference_qpos','action']}
def diff(a,b):
    if a.shape != b.shape: return {'shape_equal':False}
    mask=np.isfinite(a)&np.isfinite(b)
    delta=np.abs(a.astype(float)-b.astype(float))
    changed=~np.isclose(a,b,atol=0,rtol=0,equal_nan=True)
    rows=np.flatnonzero(changed.reshape(a.shape[0],-1).any(1))
    return {'shape_equal':True,'numeric_exact_equal':bool(np.array_equal(a,b,equal_nan=True)),
            'max_abs_difference':float(delta[mask].max()) if mask.any() else 0,
            'first_differing_step':int(rows[0]) if len(rows) else None}
def gate(mode):
    folder=ROOT/'runs'/mode;tracking=read(folder/'tracking-assessment.json');contact=read(folder/'contact-review.json')
    samples=read(folder/'contact-samples.json')[0]['samples'];contract=read(ROOT/'contact-acceptance-contract.json')
    window=round(contract['reference_expected_contact_time_tolerance_seconds']/.02);unexpected=0
    for i,record in enumerate(samples):
        expected={body for near in samples[max(0,i-window):i+window+1] for body in near['reference_ground_bodies']}
        actual={body for body in record['actual_ground_bodies'] if any(word in body.lower() for word in ['head','torso','pelvis'])}
        unexpected+=bool(actual-expected)
    minimum=contact['clips'][0]['actual_min_after_initial_100ms_m']
    return {'tracking_passed':tracking['clips'][0]['tracking_quality_passed'],'unexpected_trunk_contact_samples':unexpected,
            'actual_min_after_initial_100ms_m':minimum,
            'passed':bool(tracking['clips'][0]['tracking_quality_passed'] and unexpected<=contract['unexpected_head_torso_or_pelvis_ground_contact_samples_max'] and minimum>=-contract['actual_ground_penetration_after_initial_100ms_max_m']),
            'contact_distance_series':[[x['time_s'],x['actual_min_contact_distance_m']] for x in samples]}
def main():
    modes=[f'teacher8808-repro1031c-r{i}' for i in range(3)]
    historical=['qualified-expert-v2-teacher-raw-cuda-8808','qualified-expert-v3-teacher-raw-cuda-8808']
    traces={m:arrays(m) for m in historical+modes};records=[]
    for mode in modes:
        folder=ROOT/'runs'/mode;summary=read(folder/'summary.json')
        records.append({'mode':mode,'gate':gate(mode),'initial_policy':read(folder/'initial-policy-inputs.json'),
                        'summary':summary,'trace_sha256':sha(folder/'trajectory.npz')})
    comparisons=[]
    for a,b in itertools.combinations(historical+modes,2):
        comparisons.append({'a':a,'b':b,'reference':diff(traces[a]['reference_qpos'],traces[b]['reference_qpos']),
                            'initial_state':diff(traces[a]['state'][:1],traces[b]['state'][:1]),
                            'first_action':diff(traces[a]['action'][:1],traces[b]['action'][:1]),
                            'state':diff(traces[a]['state'],traces[b]['state']),
                            'action':diff(traces[a]['action'],traces[b]['action'])})
    result={'status':'all_fixed3_raw_seed0_fresh_process_diagnostics_complete','new_training_updates':0,
            'repetitions_requested':3,'repetitions_completed':3,'seeds':[0,0,0],'all_failures_retained':True,
            'raw_teacher_passes':sum(x['gate']['passed'] for x in records),'trials':records,'comparisons':comparisons,
            'history_hashes_equal_among_repeats':len({x['initial_policy']['initial_history']['sha256'] for x in records})==1,
            'goal_hashes_equal_among_repeats':len({x['initial_policy']['initial_goal']['sha256'] for x in records})==1,
            'all_references_equal':all(x['reference']['numeric_exact_equal'] for x in comparisons),
            'all_initial_states_equal':all(x['initial_state']['numeric_exact_equal'] for x in comparisons),
            'thresholds_changed':False,'candidate_adopted':False,'teacher_promoted':False,'full_goal_complete':False,
            'automatic_training_allowed':False}
    (ROOT/'evidence/teacher8808-repro1031c-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'passes':result['raw_teacher_passes'],'updates':0}),flush=True)
if __name__=='__main__': main()
