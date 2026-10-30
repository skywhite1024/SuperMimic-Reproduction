"""Render saved physical trajectories, with an explicitly offset reference and raw error HUD."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import mujoco
from runtime import ROOT, UNI
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import comparison_xml


def publication_font(size):
    """Use an installed font without machine-specific absolute paths."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',default='baseline');p.add_argument('--seed',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();run=ROOT/'runs'/args.mode
    trajectory=np.load(run/'trajectory.npz');summary=json.loads((run/'summary.json').read_text());manifest=json.loads((ROOT/summary.get('reference_set','references')/f'seed-{args.seed}'/'manifest.json').read_text())
    idx=next(i for i,row in enumerate(summary['trials']) if row['seed']==args.seed)
    states=trajectory['state'][:,idx];ref=trajectory['reference_qpos'][:,idx]
    reference=np.concatenate([ref[:,:3],Rotation.from_matrix(ref[:,3:12].reshape(-1,3,3)).as_quat(scalar_first=True),ref[:,12:]],1)
    policy=np.concatenate([states[:,:7],states[:,13:42]],1)
    local=ROOT/'visual/g1'
    if not local.exists():shutil.copytree(UNI/'src/unisupermimic/assets/robots/g1',local)
    xml=comparison_xml(local/'scene_supermimic.xml').replace('0.35 0.65 1.0 0.25','0.12 0.85 0.25 0.9')
    model_file=local/'continuous_comparison.xml';model_file.write_text(xml)
    model=mujoco.MjModel.from_xml_path(str(model_file));model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
    model.vis.headlight.ambient[:]=.7;model.vis.headlight.diffuse[:]=.7
    data=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.distance=5.6;camera.azimuth=135;camera.elevation=-18
    renderer=mujoco.Renderer(model,height=720,width=1280)
    font=publication_font(19);small=publication_font(16)
    videos=ROOT/'videos';videos.mkdir(exist_ok=True);out=videos/f'{args.mode}-seed{args.seed}{"-preview" if args.limit else ""}.mp4'
    ghost_offset=np.array([0.,1.5,0.])
    whole=np.concatenate([policy[:,:2],reference[:,:2]],0);lower=whole.min(0)-.5;span=np.maximum(whole.max(0)-lower+.5,1)
    scale=min(220/span[0],180/span[1]);offset=np.array([22.,0.])
    def xy(x):
        q=(x-lower)*scale+offset;q=q.copy();q[...,1]=300-q[...,1]
        return [tuple(v) for v in q]
    samples=list(range(0,round(summary['trials'][idx]['planned_seconds']*50),2))
    if args.limit:samples=samples[:args.limit]
    frames=0
    try:
        with imageio.get_writer(out,fps=25,codec='libx264',quality=8,macro_block_size=None,ffmpeg_params=['-pix_fmt','yuv420p']) as writer:
            for f in samples:
                data.qpos[:36]=policy[f];data.qpos[36:]=reference[f];data.qpos[36:39]+=ghost_offset
                mujoco.mj_kinematics(model,data)
                camera.lookat[:]=.5*(policy[f,:3]+reference[f,:3]+ghost_offset);camera.lookat[2]=.75
                renderer.update_scene(data,camera=camera)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,1280,56),fill=(18,29,42));draw.text((16,8),f'SuperMimic G1 | {args.mode} | seed {args.seed} | t={f/50:6.2f}s',font=font,fill='white')
                draw.text((16,32),'CPU MuJoCo physical rollout | green reference display offset +1.5 m in Y; metrics use unshifted world coordinates',font=small,fill='#c8d7e6')
                draw.rectangle((12,72,276,362),fill=(248,250,252));draw.text((22,82),'Fixed-world XY (metres)',font=font,fill='#182b40')
                draw.line(xy(reference[::10,:2]),fill='#a8d9b0',width=2)
                if f>0:draw.line(xy(policy[:f+1:5,:2]),fill='#c63b37',width=3)
                a=xy(reference[f:f+1,:2])[0];b=xy(policy[f:f+1,:2])[0]
                for point,color in [(a,'#008c45'),(b,'#d43333')]:draw.ellipse((point[0]-3,point[1]-3,point[0]+3,point[1]+3),fill=color)
                draw.text((22,317),f'XY: {trajectory["xy_error"][f,idx]:.3f} m',font=font,fill='#172b42')
                draw.text((22,339),f'Local body: {trajectory["body_local"][f,idx]*100:.1f} cm',font=small,fill='#172b42')
                segment=max((r for r in manifest['segments'] if r['start_frame']<=f),key=lambda r:r['segment'])
                blending=f<segment['blend_end_frame'];fail=int(np.sum(trajectory['hard_failure'][:f+1,idx] & ~np.r_[False,trajectory['hard_failure'][:f,idx]]))
                draw.rectangle((0,648,1280,720),fill=(18,29,42))
                draw.text((16,654),f'Segment {segment["segment"]}/29: {segment["source"]} | {"BLENDING" if blending else "TRACKING"} | resets: 0 | hard-failure entries: {fail}',font=font,fill='#e5f3ff')
                draw.text((16,682),f'Global body: {trajectory["body_global"][f,idx]:.3f} m | yaw: {np.rad2deg(trajectory["yaw_error"][f,idx]):.1f} deg | no online reference placement',font=small,fill='#c8d7e6')
                writer.append_data(np.asarray(image));frames+=1
                if frames in [1,1000,2000,3000]:
                    image.save(ROOT/'evidence'/f'{args.mode}-seed{args.seed}-frame{f:04d}.jpg',quality=90)
                    print('RENDER_PROGRESS',frames,flush=True)
    finally:renderer.close()
    print('PHYSICAL_TRAJECTORY_VIDEO_READY '+json.dumps({'path':str(out),'frames':frames,'seconds':frames/25}),flush=True)


if __name__=='__main__':main()
