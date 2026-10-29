"""Retained full 120 s pose contact audit; geometry only, no force reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from runtime import ROOT, make_env, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    folder = ROOT / 'runs' / args.run
    target = folder / 'continuous-contact-review.json'
    assert not target.exists()
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf8'))
    contract_file = ROOT / 'contact-acceptance-contract.json'
    contract = json.loads(contract_file.read_text(encoding='utf8'))
    with np.load(folder / 'trajectory.npz') as archive:
        states, reference = archive['state'], archive['reference_qpos']
    assert states.shape[:2] == reference.shape[:2] == (6001, 3)
    assert np.isfinite(states).all() and np.isfinite(reference).all()
    env = make_env(references('new')[:1], observation_dim=99)
    rows = []
    try:
        model = env.backend.model
        scratch = mujoco.MjData(model)
        floors = set(np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) & (model.geom_bodyid == 0)).tolist())
        def contact(pose):
            scratch.qpos[:] = pose
            mujoco.mj_fwdPosition(model, scratch)
            bodies, minimum = set(), 0.
            for item in scratch.contact:
                first, second = int(item.geom1), int(item.geom2)
                if (first in floors) == (second in floors) or item.dist > .001:
                    continue
                geom = second if first in floors else first
                bodies.add(model.body(int(model.geom_bodyid[geom])).name)
                minimum = min(minimum, float(item.dist))
            return sorted(bodies), minimum
        for index, trial in enumerate(summary['trials']):
            state, ref = states[:, index], reference[:, index]
            actual_pose = np.concatenate([state[:, :7], state[:, 13:42]], axis=1)
            reference_pose = np.concatenate([ref[:, :3], Rotation.from_matrix(ref[:, 3:12].reshape(-1, 3, 3)).as_quat(scalar_first=True), ref[:, 12:]], axis=1)
            actual = [contact(pose) for pose in actual_pose]
            desired = [contact(pose) for pose in reference_pose]
            radius = round(contract['reference_expected_contact_time_tolerance_seconds'] / .02)
            unexpected = []
            for frame, (bodies, _) in enumerate(actual):
                expected = {body for near, _ in desired[max(0, frame-radius):frame+radius+1] for body in near}
                trunk = {body for body in bodies if any(word in body.lower() for word in ['head', 'torso', 'pelvis'])}
                if trunk - expected:
                    unexpected.append({'time_s': frame*.02, 'bodies': sorted(trunk-expected)})
            minimum = min(distance for _, distance in actual[5:])
            checks = {'no_unexpected_trunk_contact': len(unexpected) <= contract['unexpected_head_torso_or_pelvis_ground_contact_samples_max'],
                      'penetration_within_engineering_limit': minimum >= -contract['actual_ground_penetration_after_initial_100ms_max_m']}
            row = {'seed': trial['seed'], 'full_seconds': 120, 'samples': 6001,
                   'actual_min_after_initial_100ms_m': minimum,
                   'unexpected_trunk_contact_samples': len(unexpected), 'unexpected_trunk_contact_events': unexpected,
                   'contact_checks': checks, 'contact_review_passed': all(checks.values()),
                   'all_pose_contact_samples': [{'time_s': i*.02, 'actual_ground_bodies': a[0], 'actual_min_contact_distance_m': a[1],
                                                'reference_ground_bodies': r[0], 'reference_min_contact_distance_m': r[1]}
                                               for i, (a, r) in enumerate(zip(actual, desired))]}
            rows.append(row)
            print('SEQUENCE_CONTACT_COMPLETE ' + json.dumps({key: value for key, value in row.items() if key not in ['all_pose_contact_samples', 'unexpected_trunk_contact_events']}), flush=True)
        result = {'run': args.run, 'checkpoint_sha256': summary['checkpoint_sha256'], 'denominator': 3,
                  'reference_version': summary['reference_version'], 'controller_variant': summary.get('controller_variant', 'native-unbounded'),
                  'contact_contract_sha256': hashlib.sha256(contract_file.read_bytes()).hexdigest(),
                  'contact_review_passed': sum(row['contact_review_passed'] for row in rows),
                  'method': 'Every 20 ms retained pose evaluated on independent scratch MuJoCo data; simulation and reference states unchanged.',
                  'contact_force_measured': False, 'new_audit_retrospective_for_unbounded_stage3': not summary.get('controller_variant'),
                  'trials': rows}
        target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    finally:
        env.close()
        env.backend.close()


if __name__ == '__main__':
    main()
