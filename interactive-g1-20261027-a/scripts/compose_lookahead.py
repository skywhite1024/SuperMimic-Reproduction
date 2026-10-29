"""Offline reference planning across the entire fixed motion order, without robot feedback."""
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
from compose_sequences import resample, FPS
from runtime import ROOT, UNI


def features(a):
    r=Rotation.from_quat(a[:,3:7]).as_matrix()
    v=np.gradient(a[:,:3],1/FPS,axis=0)
    yaw=np.arctan2(r[:,1,0],r[:,0,0])
    local_v=np.stack((np.cos(yaw)*v[:,0]+np.sin(yaw)*v[:,1],-np.sin(yaw)*v[:,0]+np.cos(yaw)*v[:,1],v[:,2]),1)
    return np.concatenate((a[:,7:]*np.sqrt(np.r_[np.full(12,3.),np.ones(17)]/29),
                           np.gradient(a[:,7:],1/FPS,axis=0)*np.sqrt(.02/29),
                           a[:,2:3]*10,r[:,2,:]*2,local_v*.5),1)


def compose(seed):
    parent=json.loads((ROOT/'references'/f'seed-{seed}/manifest.json').read_text())
    segments=parent['segments'];blend=30
    directory=UNI/'src/unisupermimic/assets/motions/g1/lafan1_supermimic'
    sources=[resample(directory/s['source']) for s in segments]
    starts=[];incoming=[];outgoing=[]
    for i,(a,s) in enumerate(zip(sources,segments)):
        maximum=len(a)-s['frames']-1
        c=np.unique(np.rint(np.linspace(0,maximum,min(160,maximum//5+1))).astype(int)) if i else np.array([0])
        starts.append(c);f=features(a)
        # Five equally spaced poses over the overlap, including vertical velocity and tilt.
        sample=np.linspace(0,blend-1,5).astype(int)
        incoming.append(f[c[:,None]+sample].reshape(len(c),-1)/np.sqrt(5))
        outgoing.append(f[c[:,None]+s['frames']-blend+sample].reshape(len(c),-1)/np.sqrt(5))
    cost=np.zeros(1);parents=[]
    for i in range(1,len(segments)):
        x,y=outgoing[i-1],incoming[i]
        transition=np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None]-2*x@y.T,0)
        total=cost[:,None]+transition
        back=np.argmin(total,axis=0);cost=total[back,np.arange(len(back))];parents.append(back)
    path=[int(np.argmin(cost))]
    for back in parents[::-1]:path.append(int(back[path[-1]]))
    path=path[::-1];result=None;rows=[]
    for i,(a,s,k) in enumerate(zip(sources,segments,path)):
        offset=int(starts[i][k]);b=a[offset:offset+s['frames']].copy();r=Rotation.from_quat(b[:,3:7])
        if result is None:
            b[:,:2]-=b[0,:2];result=b;boundary=0
        else:
            boundary=len(result)-blend;anchor=result[boundary]
            ar=Rotation.from_quat(anchor[3:7]).as_matrix();rr=r.as_matrix()[0]
            turn=Rotation.from_euler('z',np.arctan2(ar[1,0],ar[0,0])-np.arctan2(rr[1,0],rr[0,0]))
            b[:,:3]=turn.apply(b[:,:3]-b[0,:3])+[anchor[0],anchor[1],b[0,2]];b[:,3:7]=(turn*r).as_quat()
            t=np.linspace(0,1,blend)[:,None];w=6*t**5-15*t**4+10*t**3
            merged=result[-blend:]*(1-w)+b[:blend]*w
            qa,qb=result[-blend:,3:7],b[:blend,3:7]
            dot=(qa*qb).sum(1,keepdims=True);qb=np.where(dot<0,-qb,qb);theta=np.arccos(np.clip(np.abs(dot),0,1));den=np.sin(theta)
            q=np.where(den<1e-6,(1-w)*qa+w*qb,(np.sin((1-w)*theta)*qa+np.sin(w*theta)*qb)/np.maximum(den,1e-10))
            merged[:,3:7]=q/np.linalg.norm(q,axis=1,keepdims=True)
            result=np.concatenate((result[:-blend],merged,b[blend:]))
        rows.append({**s,'source_start_frame_50hz':offset,'start_frame':boundary,'blend_end_frame':boundary+(blend if i else 0),'end_frame':len(result)-1})
    assert result.shape==(6001,36) and np.isfinite(result).all()
    result=np.concatenate((result,np.repeat(result[-1:],25,axis=0)))
    out=ROOT/'references-lookahead'/f'seed-{seed}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'manifest.json').exists(),'Preserve the frozen reference'
    np.savetxt(out/'sequence.csv',result,delimiter=',',fmt='%.9g')
    metadata={**parent,'segments':rows,'reference_csv_sha256':hashlib.sha256((out/'sequence.csv').read_bytes()).hexdigest(),
              'window_selection':'Whole-sequence dynamic programming over at most 160 uniformly spaced windows per source. Five overlap poses: weighted joint position, joint velocity, root height, root tilt and heading-local root velocity. Weight scales fixed in compose_lookahead.py.',
              'planner_cost':float(cost[path[-1]]),'planner_revision':'lookahead-v1','physics_used_in_reference_selection':False,
              'original_source_order_and_segment_lengths_preserved':True}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return str(out/'sequence.csv')


if __name__=='__main__':print(json.dumps([compose(s) for s in range(6)]))
