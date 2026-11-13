"""Render audited saved poses against their reference; never resimulate physics.

Produces three fixed 12-second excerpts, including the only failed clip, and
one midpoint view for every clip. Scores/terminations remain those of the full
original evaluation. The native G1 XML/meshes supply visual geometry only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','motions','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import mujoco
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML,get_spec

    started=time.monotonic()
    result=json.loads((args.run/'result.json').read_bytes())
    protocol=json.loads((args.run/'protocol.json').read_bytes())
    assert result['clips']==40 and result['successes']==39 and result['failures']==1
    assert result['manual_reset_version']==2 and result['initial_reference_state_verified_for_all_clips']
    packed={name:sha(args.motions/name) for name in protocol['motion_files']}
    assert packed==protocol['motion_files']
    names=json.loads((args.motions/'motion_names.json').read_bytes())
    joints=json.loads((args.motions/'joint_names.json').read_bytes())
    ends=np.load(args.motions/'motion_ends.npy');starts=np.r_[0,ends[:-1]]
    pos=np.load(args.motions/'root_pos_w.npy',mmap_mode='r')
    quat=np.load(args.motions/'root_quat_w.npy',mmap_mode='r')
    q=np.load(args.motions/'joint_pos.npy',mmap_mode='r')
    assert len(names)==40 and len(joints)==29
    spec=get_spec()
    spec.worldbody.add_geom(name='render_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[50,50,.1],rgba=[.94,.95,.97,1])
    spec.worldbody.add_light(pos=[0,0,4],dir=[0,0,-1],diffuse=[.8,.8,.8])
    model=spec.compile();data=mujoco.MjData(model)
    assert model.nq==36 and model.nv==35 and model.njnt==30
    assert model.jnt_type[0]==mujoco.mjtJoint.mjJNT_FREE
    addresses=np.array([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)]
                        for name in joints])
    assert len(set(addresses))==29 and set(addresses)==set(range(7,36))
    args.output.mkdir(parents=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    camera=mujoco.MjvCamera();camera.azimuth=135;camera.elevation=-15
    renderer=mujoco.Renderer(model,height=480,width=640)
    ledger=[];videos=[]
    sheet=Image.new('RGB',(2400,1664),'#101820')
    ImageDraw.Draw(sheet).text((20,16),'All 40 clips | one saved midpoint each | policy left, reference right | full outcomes shown',font=font,fill='white')

    def reference(index,frame):
        k=int(starts[index])+frame
        return np.r_[pos[k],quat[k],q[k]].astype(np.float64)

    def render_pair(actual,target):
        camera.lookat[:]=(actual[:3]+target[:3])/2
        camera.lookat[2]=.75
        camera.distance=3+float(np.linalg.norm(actual[:2]-target[:2]))
        images=[]
        for state in (actual,target):
            data.qpos[:7]=state[:7];data.qpos[addresses]=state[7:36]
            data.qvel[:]=0
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera)
            images.append(Image.fromarray(renderer.render().copy()))
        out=Image.new('RGB',(1280,480));out.paste(images[0],(0,0));out.paste(images[1],(640,0))
        return out

    try:
        for record in result['records']:
            i=record['index'];assert names[i]==record['name']
            trajectory=args.run/record['trajectory']
            with np.load(trajectory) as archive:
                states=archive['full_state']
            assert states.dtype==np.float32 and states.shape==(record['steps'],71) and np.isfinite(states).all()
            assert np.allclose(states[0,:7],reference(i,0)[:7],atol=1e-5,rtol=1e-5)
            ledger.append({'index':i,'name':record['name'],'steps':len(states),
                           'trajectory_bytes':trajectory.stat().st_size,'trajectory_sha256':sha(trajectory)})
            midpoint=(len(states)-1)//2
            tile=Image.new('RGB',(480,200),'#101820')
            tile.paste(render_pair(states[midpoint],reference(i,midpoint)).resize((480,180)),(0,20))
            outcome='SUCCESS' if record['success'] else 'FAIL: '+','.join(record['termination_terms'])
            ImageDraw.Draw(tile).text((4,1),f'{i:02d} {record["name"]} | {outcome}',font=small,
                                      fill='#78d9a0' if record['success'] else '#ff9393')
            sheet.paste(tile,((i%5)*480,64+(i//5)*200))
            if i not in (8,23,28):
                continue
            if i==8:
                center=int(np.argmin(pos[int(starts[i]):int(ends[i]),2]))
                start=max(0,min(center-300,len(states)-600))
                reason='Reference minimum root height centered where possible'
            elif i==23:
                start=len(states)-1-2*299
                reason='Last12s of the onlyfailedfullclip; finalrenderedstate precedestheterminatingstep'
            else:
                start=0;reason='First12s of the predeterminedwalkingclip'
            frames=list(range(start,start+600,2));assert len(frames)==300 and frames[-1]<len(states)
            path=args.output/f'{i:02d}-{record["name"]}.mp4'
            writer=imageio_ffmpeg.write_frames(str(path),size=(1280,576),fps=25,
                codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',
                output_params=['-crf','22','-preset','fast'])
            writer.send(None)
            for ordinal,frame in enumerate(frames):
                target=reference(i,frame);actual=states[frame]
                canvas=Image.new('RGB',(1280,576),'#101820')
                canvas.paste(render_pair(actual,target),(0,64))
                draw=ImageDraw.Draw(canvas)
                draw.text((16,7),f'{record["name"]} | Full clip: {outcome} | excerpt t={frame*.02:.2f}s',font=font,fill='white')
                draw.text((16,36),'Saved policy trajectory (pre-action pose)',font=small,fill='#8bc9ff')
                draw.text((656,36),'Reference motion (unclipped target pose)',font=small,fill='#78d9a0')
                root_error=float(np.linalg.norm(actual[:3]-target[:3]))
                joint_rms=float(np.sqrt(np.mean((actual[7:36]-target[7:36])**2))*180/np.pi)
                draw.text((16,550),f'Pose rendering only; no physics rerun | root distance {root_error:.3f}m | joint RMS {joint_rms:.1f}deg | 1x,25fps',font=small,fill='#d3dce4')
                writer.send(np.asarray(canvas))
                if ordinal in (0,150,299):
                    canvas.save(args.output/f'{i:02d}-preview-{ordinal:03d}.png')
            writer.close()
            decoded=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
            metadata=next(decoded);count=sum(1 for _ in decoded)
            assert metadata['size']==(1280,576) and abs(metadata['fps']-25)<1e-6 and count==300
            videos.append({'index':i,'name':record['name'],'file':path.name,'bytes':path.stat().st_size,
                'sha256':sha(path),'frames':count,'fps':25,'duration_seconds':count/25,
                'simulation_frame_indices':frames,'selection':reason,'full_clip_success':record['success'],
                'full_clip_termination_terms':record['termination_terms']})
            print('SAVED_TRAJECTORY_VIDEO '+json.dumps({k:v for k,v in videos[-1].items() if k!='simulation_frame_indices'}),flush=True)
    finally:
        renderer.close()
    sheet_path=args.output/'all40-midpoints.png';sheet.save(sheet_path)
    receipt={'status':'saved_trajectory_rendering_completed','source_sha256':sha(Path(__file__)),
        'result_sha256':sha(args.run/'result.json'),'protocol_sha256':sha(args.run/'protocol.json'),
        'packed_file_sha256':packed,'native_g1_visual_assets':{str(p.relative_to(G1_XML.parent)):sha(p)
            for p in [G1_XML,*sorted((G1_XML.parent/'assets').iterdir())] if p.is_file()},
        'all40_trajectory_ledger':ledger,'videos':videos,
        'sheet':{'file':sheet_path.name,'bytes':sheet_path.stat().st_size,'sha256':sha(sheet_path)},
        'mujoco_version':mujoco.__version__,'elapsed_seconds':time.monotonic()-started,
        'joint_qpos_addresses':addresses.tolist(),'camera':'Samecamera perpair,midpointroottracking,azimuth135,elevation-15',
        'visual_scene_changes':['Addedfloorandlightonly'],'physics_resimulated':False,
        'full_clip_scores_recomputed_here':False,'full_clip_outcomes':{'successes':39,'failures':1},
        'video_excerpts_are_not_full_clip_evaluations':True,'new_policy_quality_claimed':False}
    (args.output/'rendering.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('SAVED_TRAJECTORY_RENDERING_COMPLETED '+json.dumps({'videos':len(videos),'midpoints':40,'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
