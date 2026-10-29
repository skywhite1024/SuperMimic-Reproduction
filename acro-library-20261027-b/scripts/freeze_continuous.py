"""Pre-freeze three 120 s reference schedules covering every complete new clip."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'implementation/uni/src/unisupermimic/assets/motions/g1/lafan1_supermimic'
FPS=50;BRIDGE=40


def resample(path):
    raw=np.loadtxt(path,delimiter=',');times=np.arange(len(raw))/30;target=np.arange(int(np.floor(times[-1]*FPS))+1)/FPS
    result=np.empty((len(target),36));result[:,:3]=np.stack([np.interp(target,times,raw[:,j]) for j in range(3)],1)
    result[:,3:7]=Slerp(times,Rotation.from_quat(raw[:,3:7]))(target).as_quat()
    result[:,7:]=np.stack([np.interp(target,times,raw[:,j]) for j in range(7,36)],1)
    return result


def append(result,incoming,rows,metadata):
    incoming=incoming.copy();first=Rotation.from_quat(incoming[0,3:7]).as_matrix();last=Rotation.from_quat(result[-1,3:7]).as_matrix()
    turn=Rotation.from_euler('z',np.arctan2(last[1,0],last[0,0])-np.arctan2(first[1,0],first[0,0]))
    origin=incoming[0,:3].copy();incoming[:,:3]=turn.apply(incoming[:,:3]-origin)+[result[-1,0],result[-1,1],origin[2]]
    incoming[:,3:7]=(turn*Rotation.from_quat(incoming[:,3:7])).as_quat()
    t=np.arange(1,BRIDGE+1)/BRIDGE;weight=(6*t**5-15*t**4+10*t**3)[:,None]
    bridge=result[-1:]*(1-weight)+incoming[:1]*weight
    bridge[:,3:7]=Slerp([0,1],Rotation.from_quat(np.stack([result[-1,3:7],incoming[0,3:7]])))(weight[:,0]).as_quat()
    boundary=len(result)-1;result=np.concatenate([result,bridge,incoming[1:]])
    rows.append({**metadata,'transition_start_frame':boundary,'source_start_in_sequence_frame':boundary+BRIDGE,
                 'end_frame':len(result)-1,'source_frames_50hz':len(incoming),'bridge_frames':BRIDGE})
    return result


def main():
    dest=ROOT/'continuous-frozen-v1';assert not dest.exists(),'Preserve all frozen schedules; use a new version for changes'
    manifest=json.loads((ROOT/'library-manifest.json').read_text());new={}
    for row in manifest['files']:
        path=ROOT/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];new[path.stem]=(path,resample(path))
    groups=[[] for _ in range(3)];lengths=[0]*3
    for name in sorted(new,key=lambda name:len(new[name][1]),reverse=True):
        i=int(np.argmin(lengths));groups[i].append(name);lengths[i]+=len(new[name][1])-1
    old={family:sorted(OLD.glob(family+'*.csv')) for family in ['walk','run','dance','jumps']};cache={}
    stand=resample(old['walk'][0])[0].copy();stand[:2]=0;receipts=[]
    for seed,names in enumerate(groups):
        rng=np.random.default_rng(seed);rng.shuffle(names);result=np.repeat(stand[None],FPS+1,axis=0);rows=[]
        def add_old(family,steps):
            nonlocal result
            path=old[family][int(rng.integers(len(old[family])))];source=cache.setdefault(str(path),resample(path))
            candidates=np.arange(0,len(source)-steps-1,10);assert len(candidates)
            pose=((source[candidates,7:]-result[-1,7:])**2).mean(1)+.2*(source[candidates,2]-result[-1,2])**2
            offset=int(candidates[np.argmin(pose)]);incoming=source[offset:offset+steps+1]
            result=append(result,incoming,rows,{'source':path.stem,'group':'old_lafan','family':family,'source_start_frame_50hz':offset,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for i,name in enumerate(names):
            add_old(['walk','dance','run','jumps'][i%4],150)
            path,source=new[name];result=append(result,source,rows,{'source':name,'group':'new_raw_full_clip','source_start_frame_50hz':0,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_temporal_crop':False})
        final_budget=BRIDGE+3*FPS
        for slot in range(40):
            remaining=120*FPS+1-len(result)-final_budget
            if remaining<=BRIDGE:break
            add_old(['walk','run','dance','jumps'][slot%4],min(150,remaining-BRIDGE))
        final_pose=stand.copy();final_pose[:2]=result[-1,:2]
        hold_frames=120*FPS+1-len(result)-BRIDGE;assert hold_frames>=150
        result=append(result,np.repeat(final_pose[None],hold_frames+1,axis=0),rows,{'source':'fixed_upright_terminal_hold','group':'terminal_hold','source_start_frame_50hz':0})
        assert result.shape==(120*FPS+1,36) and np.isfinite(result).all()
        assert max(abs(np.linalg.norm(result[:,3:7],axis=1)-1))<1e-5
        result=np.concatenate([result,np.repeat(result[-1:],25,axis=0)])
        directory=dest/f'seed-{seed}';directory.mkdir(parents=True);file=directory/'sequence.csv';np.savetxt(file,result,delimiter=',',fmt='%.9g')
        receipt={'seed':seed,'seconds':120,'fps':FPS,'reference_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'frames_including_future_hold':len(result),
                 'new_full_clip_ids':names,'segments':rows,'initial_hold_s':1,'terminal_hold_s':hold_frames/FPS,
                 'alignment':'fixed reference-only yaw and XY transform at incoming segment start; no online robot placement',
                 'bridge':'0.8 second extra quintic pose/joint interpolation and shortest-arc quaternion SLERP; raw new clips are not overlapped or temporally cropped',
                 'bridge_dynamic_feasibility_verified':False,'robot_execution_status':'not_evaluated','original_raw_library_retained':True}
        (directory/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt)
    all_names=[name for receipt in receipts for name in receipt['new_full_clip_ids']];assert sorted(all_names)==sorted(new) and len(all_names)==15
    index={'frozen_at':datetime.now(timezone.utc).isoformat(),'seeds':[0,1,2],'new_denominator':15,'new_full_clips_in_union':all_names,
           'source_commit':manifest['commit'],'candidate_policy_not_used_for_planning':True,'robot_execution_status':'not_evaluated','sequences':receipts}
    (dest/'index.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps({'sequences':len(receipts),'new_union':len(all_names),'segments':[len(x['segments']) for x in receipts],'seconds_each':120}))


if __name__=='__main__':main()
