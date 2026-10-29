"""Versioned paired test: same frozen schedule, only the two existing Z calibrations."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    original=ROOT/'continuous-frozen-v1/index.json';index=json.loads(original.read_text())
    variant_path=ROOT/'ground-calibration-v1/manifest.json';variant=json.loads(variant_path.read_text())
    calibrations={row['clip']:row for row in variant['files'] if row['constant_root_z_shift_m']}
    assert set(calibrations)=={'90_14_stageii_unitree_g1','90_15_stageii_unitree_g1'}
    destination=ROOT/'continuous-corrected-v1';assert not destination.exists()
    receipts=[]
    for sequence in index['sequences']:
        seed=sequence['seed'];source=ROOT/f'continuous-frozen-v1/seed-{seed}/sequence.csv'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==sequence['reference_sha256']
        lines=source.read_text().splitlines();before=np.loadtxt(source,delimiter=',');shift=np.zeros(len(lines));modified=[]
        for segment in sequence['segments']:
            if segment['source'] not in calibrations:continue
            row=calibrations[segment['source']];delta=row['constant_root_z_shift_m']
            boundary=segment['transition_start_frame'];start=segment['source_start_in_sequence_frame'];end=segment['end_frame'];bridge=segment['bridge_frames']
            t=np.arange(1,bridge+1)/bridge;weight=6*t**5-15*t**4+10*t**3
            shift[boundary+1:start]+=delta*weight[:-1]
            shift[start:end+1]+=delta
            shift[end+1:end+bridge+1]+=delta*(1-weight[:len(shift[end+1:end+bridge+1])])
            modified.append({'source':segment['source'],'constant_clip_root_z_shift_m':delta,'source_start_frame':start,'end_frame':end,'incoming_and_outgoing_bridge_height_interpolated':True})
        changed=[]
        for line,delta in zip(lines,shift):
            if delta:
                columns=line.split(',');columns[2]=format(float(columns[2])+delta,'.12g');line=','.join(columns)
            changed.append(line)
        directory=destination/f'seed-{seed}';directory.mkdir(parents=True)
        target=directory/'sequence.csv';target.write_text('\n'.join(changed)+'\n',newline='\n')
        after=np.loadtxt(target,delimiter=',');assert np.array_equal(before[:,np.arange(36)!=2],after[:,np.arange(36)!=2])
        assert np.max(abs(after[:,2]-before[:,2]-shift))<1e-10
        receipt={**sequence,'reference_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'raw_frozen_reference_sha256':sequence['reference_sha256'],
                 'reference_variant':'continuous-corrected-v1','ground_calibration_manifest_sha256':hashlib.sha256(variant_path.read_bytes()).hexdigest(),
                 'only_root_z_and_bridge_z_modified':True,'XY_yaw_joint_pose_order_timing_unchanged':True,'changes':modified}
        (directory/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt)
    result={**index,'frozen_at':datetime.now(timezone.utc).isoformat(),'sequences':receipts,'reference_variant':'continuous-corrected-v1',
            'raw_frozen_index_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'policy_outputs_used_for_reference_alignment':False,
            'training_uses_these_sequences':False,'raw_frozen_test_preserved_and_reported_separately':True,'new_clip_order_and_timing_unchanged':True}
    (destination/'index.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'version':'continuous-corrected-v1','sequences':3,'calibrated_full_clips':sum(len(x['changes']) for x in receipts),'all_other_columns_exact':True}),flush=True)

if __name__=='__main__':main()
