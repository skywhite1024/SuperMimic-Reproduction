"""Reconstruct frozen references from their manifests and original pinned input files."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'implementation/uni/src/unisupermimic/assets/motions/g1/lafan1_supermimic'
CACHE={}
def resample(path):
    path=path.resolve()
    if path in CACHE:return CACHE[path]
    raw=np.loadtxt(path,delimiter=',');t=np.arange(len(raw))/30;target=np.arange(int(np.floor(t[-1]*50))+1)/50
    a=np.empty((len(target),36));a[:,:3]=np.stack([np.interp(target,t,raw[:,i]) for i in range(3)],1)
    a[:,3:7]=Slerp(t,Rotation.from_quat(raw[:,3:7]))(target).as_quat()
    a[:,7:]=np.stack([np.interp(target,t,raw[:,i]) for i in range(7,36)],1)
    CACHE[path]=a;return a
def save(path,a,expected,fmt):
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,f'Preserve conflicting reference: {path}'
        return
    temp=path.with_suffix('.generated');np.savetxt(temp,a,delimiter=',',fmt=fmt)
    actual=hashlib.sha256(temp.read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError(f'Reference hash differs for {path.relative_to(ROOT)}; generated output preserved: {actual}')
    temp.replace(path)
def basic(folder):
    lookahead=folder.name=='references-lookahead'
    for file in sorted(folder.glob('seed-*/manifest.json')):
        m=json.loads(file.read_text());result=None;blend=round(m['blend_seconds']*50)
        for row in m['segments']:
            source=OLD/row['source'];assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
            offset=row['source_start_frame_50hz'];b=resample(source)[offset:offset+row['frames']].copy();r=Rotation.from_quat(b[:,3:7])
            if result is None:
                b[:,:2]-=b[0,:2]
                if not lookahead:b[:,3:7]=r.as_quat()
                result=b;continue
            boundary=len(result)-blend;anchor=result[boundary];ar=Rotation.from_quat(anchor[3:7]).as_matrix();br=r.as_matrix()[0]
            turn=Rotation.from_euler('z',np.arctan2(ar[1,0],ar[0,0])-np.arctan2(br[1,0],br[0,0]))
            b[:,:3]=turn.apply(b[:,:3]-b[0,:3])+[anchor[0],anchor[1],b[0,2]];b[:,3:7]=(turn*r).as_quat()
            t=np.linspace(0,1,blend)[:,None];w=6*t**5-15*t**4+10*t**3
            merged=result[-blend:]*(1-w)+b[:blend]*w
            qa,qb=result[-blend:,3:7],b[:blend,3:7];dot=(qa*qb).sum(1,keepdims=True);qb=np.where(dot<0,-qb,qb)
            theta=np.arccos(np.clip(np.abs(dot),0,1));den=np.sin(theta)
            q=np.where(den<1e-6,(1-w)*qa+w*qb,(np.sin((1-w)*theta)*qa+np.sin(w*theta)*qb)/np.maximum(den,1e-10))
            merged[:,3:7]=q/np.linalg.norm(q,axis=1,keepdims=True);result=np.concatenate((result[:-blend],merged,b[blend:]))
        result=np.concatenate((result,np.repeat(result[-1:],25,axis=0)))
        save(file.with_name('sequence.csv'),result,m['reference_csv_sha256'],'%.9g')
def unified():
    directory=ROOT/'acro-library-20261027-b'
    # Original frozen manifests contain all source windows, timing and bridge lengths.
    for file in sorted((directory/'continuous-frozen-v1').glob('seed-*/manifest.json')):
        m=json.loads(file.read_text());stand=resample(sorted(OLD.glob('walk*.csv'))[0])[0].copy();stand[:2]=0;result=np.repeat(stand[None],51,axis=0)
        for row in m['segments']:
            if row['group']=='terminal_hold':
                pose=stand.copy();pose[:2]=result[-1,:2];b=np.repeat(pose[None],row['source_frames_50hz'],axis=0)
            else:
                source=(directory/'data'/(row['source']+'.csv')) if row['group']=='new_raw_full_clip' else OLD/(row['source']+'.csv')
                assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
                off=row['source_start_frame_50hz'];b=resample(source)[off:off+row['source_frames_50hz']].copy()
            first=Rotation.from_quat(b[0,3:7]).as_matrix();last=Rotation.from_quat(result[-1,3:7]).as_matrix()
            turn=Rotation.from_euler('z',np.arctan2(last[1,0],last[0,0])-np.arctan2(first[1,0],first[0,0]))
            origin=b[0,:3].copy();b[:,:3]=turn.apply(b[:,:3]-origin)+[result[-1,0],result[-1,1],origin[2]];b[:,3:7]=(turn*Rotation.from_quat(b[:,3:7])).as_quat()
            count=row['bridge_frames'];t=np.arange(1,count+1)/count;w=(6*t**5-15*t**4+10*t**3)[:,None]
            bridge=result[-1:]*(1-w)+b[:1]*w;bridge[:,3:7]=Slerp([0,1],Rotation.from_quat(np.stack([result[-1,3:7],b[0,3:7]])))(w[:,0]).as_quat()
            result=np.concatenate((result,bridge,b[1:]))
        # Final hold and length are part of the frozen protocol.
        result=result[:6001]
        if len(result)<6001:result=np.concatenate((result,np.repeat(result[-1:],6001-len(result),axis=0)))
        result=np.concatenate((result,np.repeat(result[-1:],25,axis=0)))
        save(file.with_name('sequence.csv'),result,m['reference_sha256'],'%.9g')
def calibrated():
    directory=ROOT/'acro-library-20261027-b';variant=json.loads((directory/'ground-calibration-v1/manifest.json').read_text())
    shifts={r['clip']:r['constant_root_z_shift_m'] for r in variant['files'] if r['constant_root_z_shift_m']}
    for row in variant['files']:
        target=directory/row['path'];raw=directory/'data'/(row['clip']+'.csv');target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256'];continue
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==row['raw_sha256']
        if row['constant_root_z_shift_m']:
            lines=[]
            for line in raw.read_text().splitlines():
                cols=line.split(',');cols[2]=format(float(cols[2])+row['constant_root_z_shift_m'],'.12g');lines.append(','.join(cols))
            target.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        else:target.write_bytes(raw.read_bytes())
        assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
    for file in sorted((directory/'continuous-corrected-v1').glob('seed-*/manifest.json')):
        m=json.loads(file.read_text());raw=directory/'continuous-frozen-v1'/file.parent.name/'sequence.csv'
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==m['raw_frozen_reference_sha256']
        target=file.with_name('sequence.csv')
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest()==m['reference_sha256'];continue
        lines=raw.read_text().splitlines();shift=np.zeros(len(lines))
        for segment in m['segments']:
            if segment['source'] not in shifts:continue
            delta=shifts[segment['source']];boundary=segment['transition_start_frame'];start=segment['source_start_in_sequence_frame'];end=segment['end_frame'];bridge=segment['bridge_frames']
            t=np.arange(1,bridge+1)/bridge;weight=6*t**5-15*t**4+10*t**3
            shift[boundary+1:start]+=delta*weight[:-1];shift[start:end+1]+=delta
            shift[end+1:end+bridge+1]+=delta*(1-weight[:len(shift[end+1:end+bridge+1])])
        changed=[]
        for line,delta in zip(lines,shift):
            if delta:
                cols=line.split(',');cols[2]=format(float(cols[2])+delta,'.12g');line=','.join(cols)
            changed.append(line)
        target.write_text('\n'.join(changed)+'\n',encoding='utf-8',newline='\n')
        assert hashlib.sha256(target.read_bytes()).hexdigest()==m['reference_sha256']
def main():
    for name in ['references','references-lookahead','references-velocity-matched']:basic(ROOT/'interactive-g1-20261027-a'/name)
    unified()
    calibrated()
    print('All reconstructed raw and calibrated reference files match their frozen hashes.')
if __name__=='__main__':main()
