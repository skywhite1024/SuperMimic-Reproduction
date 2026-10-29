"""Render a complete retained acrobatics trajectory; never advances or edits a simulation."""
import argparse
import hashlib
import json
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import mujoco
from runtime import ROOT, UNI
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import comparison_xml


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True);a=p.parse_args()
    run=ROOT/'runs'/a.mode;summary=json.loads((run/'summary.json').read_text())
    with np.load(run/'trajectory.npz') as traj:
        states=traj['state'];ref=traj['reference_qpos'];errors=traj['root_error'];failed=traj['hard_failure']
    assert states.ndim==2 and states.shape[1]==71 and ref.shape==(len(states),41)
    actual=np.concatenate((states[:,:7],states[:,13:42]),1)
    reference=np.concatenate((ref[:,:3],Rotation.from_matrix(ref[:,3:12].reshape(-1,3,3)).as_quat(scalar_first=True),ref[:,12:]),1)
    rzz=1-2*(states[:,4]**2+states[:,5]**2)
    local=ROOT/'visual/g1'
    if not local.exists():shutil.copytree(UNI/'src/unisupermimic/assets/robots/g1',local)
    xml=comparison_xml(local/'scene_supermimic.xml').replace('0.35 0.65 1.0 0.25','0.12 0.85 0.25 0.9')
    path=local/'acro_comparison.xml';path.write_text(xml)
    model=mujoco.MjModel.from_xml_path(str(path));model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
    model.vis.headlight.ambient[:]=.7;model.vis.headlight.diffuse[:]=.7
    data=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.distance=4.8;camera.azimuth=145;camera.elevation=-15
    renderer=mujoco.Renderer(model,height=720,width=1280)
    font=ImageFont.truetype('[local]/arial.ttf',16)
    out=ROOT/'videos'/f'{a.mode}.mp4';assert not out.exists(),'Preserve prior rendered result'
    frames=0;offset=np.array([0.,1.5,0.])
    try:
        with imageio.get_writer(out,fps=25,codec='libx264',quality=8,macro_block_size=None,ffmpeg_params=['-pix_fmt','yuv420p']) as writer:
            for i in range(0,len(states),2):
                data.qpos[:36]=actual[i];data.qpos[36:]=reference[i];data.qpos[36:39]+=offset
                mujoco.mj_kinematics(model,data)
                camera.lookat[:]=.5*(actual[i,:3]+reference[i,:3]+offset);camera.lookat[2]=.7
                renderer.update_scene(data,camera=camera);frame=Image.fromarray(renderer.render());d=ImageDraw.Draw(frame)
                d.rectangle((0,0,1280,82),fill='#122034');d.text((18,10),f'G1 physical acrobatics | {a.mode} | {i*.02:.2f}s',fill='white',font=font)
                d.text((18,42),'Gray: actual policy rollout. Green: reference, display offset +1.5 m Y. Metrics: unshifted world.',fill='#c8d7e6',font=small)
                d.rectangle((0,628,1280,720),fill='#122034')
                d.text((18,637),f'Root error {errors[i]:.3f} m | actual Rzz {rzz[i]:+.3f} | original hard-gate failure frames {int(failed[:i+1].sum())} | resets {summary["post_initialization_resets"]}',fill='white',font=font)
                d.text((18,674),f'Full clip acceptance: {"PASS" if summary["passed"] else "FAIL"} | inversion: {summary["inversion_observed"]} | final upright ending: {summary["landing_last_half_second_upright"]}',fill='#b3e6bd' if summary['passed'] else '#ffc4b5',font=font)
                writer.append_data(np.asarray(frame));frames+=1
    finally:renderer.close()
    # Complete decode, including the failed tail, before publishing a video receipt.
    reader=imageio.get_reader(out);decoded=0;sampled=[]
    indices=set(np.linspace(0,frames-1,5).round().astype(int).tolist())
    try:
        for i,frame in enumerate(reader):
            assert frame.shape==(720,1280,3);decoded+=1
            if i in indices:sampled.append(Image.fromarray(frame).resize((640,360)))
    finally:reader.close()
    assert decoded==frames and abs(decoded/25-summary['elapsed_seconds'])<.041
    sheet=Image.new('RGB',(1280,1080),'white')
    for i,frame in enumerate(sampled):sheet.paste(frame,(i%2*640,i//2*360))
    sheet.save(ROOT/'evidence'/f'{a.mode}-contact-sheet.jpg',quality=90)
    receipt={'path':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'frames':frames,'complete_decode_passed':True,'seconds':frames/25,'physical_acceptance_passed':summary['passed'],'only_retained_physical_states_rendered':True}
    (ROOT/'evidence'/f'video-{a.mode}.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
