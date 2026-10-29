"""Controlled constant-height calibration ablation; preserve all raw source bytes."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    original=json.loads((ROOT/'library-manifest.json').read_text());clearance=json.loads((ROOT/'evidence/raw-ground-clearance.json').read_text())
    distances={row['clip']:row for row in clearance['clips']};dest=ROOT/'ground-calibration-v1';assert not dest.exists()
    rows=[]
    for row in original['files']:
        source=ROOT/row['path'];assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256'];name=source.stem
        target=dest/'data'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shift=0.
        if name in ['90_14_stageii_unitree_g1','90_15_stageii_unitree_g1']:
            before=np.loadtxt(source,delimiter=',');shift=.005-float(np.median(distances[name]['first_half_second_clearance_m']))
            lines=[]
            for line in source.read_text().splitlines():
                columns=line.split(',');columns[2]=format(float(columns[2])+shift,'.12g');lines.append(','.join(columns))
            target.write_text('\n'.join(lines)+'\n',newline='\n');after=np.loadtxt(target,delimiter=',')
            assert np.array_equal(before[:,np.arange(36)!=2],after[:,np.arange(36)!=2])
            assert np.max(abs((after[:,2]-before[:,2])-shift))<1e-10
        else:target.write_bytes(source.read_bytes())
        rows.append({'path':str(target.relative_to(ROOT)).replace('\\','/'),'clip':name,'raw_sha256':row['sha256'],
                     'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'constant_root_z_shift_m':shift,
                     'all_other_columns_identical':True,'frame_count_unchanged':True,'physical_acceptance':'not_evaluated'})
    manifest={'variant':'ground-calibration-v1','created_at':datetime.now(timezone.utc).isoformat(),'source_commit':original['commit'],
              'denominator':15,'modified_clips':2,'method':'Only 90_14 and 90_15: subtract median minimum collision-geometry clearance over the first 0.5 seconds, leaving 5 mm clearance; one constant Z translation for the whole clip',
              'raw_ground_audit_sha256':hashlib.sha256((ROOT/'evidence/raw-ground-clearance.json').read_bytes()).hexdigest(),
              'raw_library_unchanged':True,'current_raw_GPU_training_changed':False,'joints_orientation_XY_timing_unchanged':True,
              'remaining_hand_support_penetration_or_dynamics_not_repaired':True,'status':'local_controlled_ablation_not_formal_acceptance','files':rows}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'denominator':15,'shifts':{r['clip']:r['constant_root_z_shift_m'] for r in rows if r['constant_root_z_shift_m']}}))


if __name__=='__main__':main()
