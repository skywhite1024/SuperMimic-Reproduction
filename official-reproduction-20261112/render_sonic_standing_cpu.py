"""Independently recompute a scalar SONIC standing trace and render its result."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve()
    result=json.loads((root/'result.json').read_bytes())
    protocol=json.loads((root/'protocol.json').read_bytes())
    assert digest(root/'protocol.json')==result['protocol_sha256']
    assert digest(root/'trace.npz')==result['trace']['sha256']
    with np.load(root/'trace.npz',allow_pickle=False) as data:
        arrays={name:data[name] for name in data.files}
    count=result['control_steps'];states=arrays['qpos']
    assert states.shape==(count+1,36) and arrays['qvel'].shape==(count+1,35)
    assert arrays['actions'].shape==(count,29) and np.isfinite(states).all()
    assert np.allclose(states[0,:7],protocol['initial_root_pose'],atol=1e-14)
    q=states[1:,3:7];assert np.allclose(np.sum(q*q,axis=1),1.,atol=1e-12)
    tilt=2*(q[:,1]**2+q[:,2]**2)
    assert np.allclose(tilt,arrays['tilt'],atol=1e-12,rtol=0)
    assert np.array_equal(states[1:,2],arrays['root_height'])
    assert float(np.min(states[1:,2]))==result['minimum_root_height']
    assert np.isclose(np.max(tilt),result['maximum_tilt'],atol=1e-12)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seconds=np.arange(1,count+1)*protocol['control_seconds']
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True)
    axes[0].plot(seconds,states[1:,2],color='#1669a5',label='Observed pelvis height')
    axes[0].axhline(.76,color='#777777',ls='--',lw=1,label='Initial/reference height')
    axes[0].set_ylabel('Height (m)');axes[0].legend(loc='upper right',fontsize=8)
    axes[1].plot(seconds,np.degrees(np.arccos(np.clip(1-tilt,-1,1))),color='#b45f12')
    axes[1].set_ylabel('Root tilt (degrees)')
    axes[2].plot(seconds,np.max(np.abs(arrays['actions']),axis=1),color='#317c3b')
    axes[2].set_ylabel('Max |raw action|');axes[2].set_xlabel('Simulated time (seconds)')
    for axis in axes:axis.grid(alpha=.2)
    fig.suptitle('Released SONIC: 10-second standing pilot on scalar MuJoCo / CPU\n'
                 'Common G1 model + SONIC gains/scales; no motion-tracking or Warp-equivalence claim',fontsize=12)
    for suffix in ('png','svg'):fig.savefig(root/('standing-pilot.'+suffix),dpi=160)
    plt.close(fig)
    image_error=None
    try:
        import mujoco
        import supermimic.tasks
        from mjlab.scene import Scene
        from mjlab.tasks.registry import load_env_cfg
        cfg=load_env_cfg('SuperMimic-GenCoord-G1',play=True);cfg.scene.num_envs=1
        model=Scene(cfg.scene,'cpu').compile();data=mujoco.MjData(model)
        data.qpos[:]=states[-1];data.qvel[:]=arrays['qvel'][-1]
        mujoco.mj_forward(model,data)
        camera=mujoco.MjvCamera();camera.lookat[:]=[float(states[-1,0]),float(states[-1,1]),.75]
        camera.distance=2.4;camera.azimuth=135;camera.elevation=-15
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            renderer.update_scene(data,camera)
            pixels=renderer.render()
        plt.imsave(root/'final-pose.png',pixels)
    except Exception as error:image_error=type(error).__name__+': '+str(error)
    report=dict(status='all_saved_scalar_standing_arrays_and_independent_height_tilt_recomputation_passed',
        result_sha256=digest(root/'result.json'),trace_sha256=digest(root/'trace.npz'),control_steps=count,
        independently_recomputed_minimum_height=float(np.min(states[1:,2])),
        independently_recomputed_maximum_tilt=float(np.max(tilt)),
        maximum_root_tilt_degrees=float(np.max(np.degrees(np.arccos(np.clip(1-tilt,-1,1))))),
        figure_files=[dict(file=p.name,sha256=digest(p)) for p in (root/'standing-pilot.png',root/'standing-pilot.svg')],
        final_pose_rendered=(root/'final-pose.png').is_file(),final_pose_render_error=image_error,
        simulation_backend='scalar MuJoCo CPU',full_motion_tracking_evaluated=False,paper_results_reproduced=False)
    (root/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
