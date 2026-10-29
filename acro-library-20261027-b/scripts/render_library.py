"""Render complete retained actual/reference trajectories, including all failed tails."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
import mujoco
from runtime import ROOT,UNI
from unisupermimic.tasks.motion_tracking.g1.supermimic.render import comparison_xml


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--clips',nargs='+');p.add_argument('--video-label');p.add_argument('--verify-existing',action='store_true');p.add_argument('--include-old',action='store_true');a=p.parse_args()
    if a.include_old:assert a.clips,'Choose explicit old clips; do not render the entire old library by accident'
    directory=ROOT/'runs'/a.run;summary=json.loads((directory/'summary.json').read_text())
    continuous='trials' in summary
    rows=[{**row,'clip':f'seed-{row["seed"]}','full_clip_steps':6000,'group':'continuous'} for row in summary['trials']] if continuous else summary['clips']
    assessment=directory/'tracking-assessment.json';quality={x['clip']:x for x in json.loads(assessment.read_text())['clips']} if assessment.exists() else {}
    with np.load(directory/'trajectory.npz') as trace:states=trace['state'];references=trace['reference_qpos'];errors=trace['xy_error'];failed=trace['hard_failure' if continuous else 'hard_failure_aligned']
    selected=[i for i,row in enumerate(rows) if (row['group'] in ['new','continuous'] or a.include_old) and (not a.clips or row['clip'] in a.clips)]
    assert selected;assets=ROOT/'visual/g1'
    if not assets.exists():shutil.copytree(UNI/'src/unisupermimic/assets/robots/g1',assets)
    xml=comparison_xml(assets/'scene_supermimic.xml').replace('0.35 0.65 1.0 0.25','0.12 0.85 0.25 0.9')
    file=assets/'comparison.xml';file.write_text(xml);model=mujoco.MjModel.from_xml_path(str(file))
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=720;model.vis.headlight.ambient[:]=.7;model.vis.headlight.diffuse[:]=.7;model.vis.map.zfar=1000
    data=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.azimuth=145;camera.elevation=-15
    font=ImageFont.truetype('[local]/arial.ttf',16)
    video_label=a.video_label or a.run;target=ROOT/'videos'/video_label;assert target.resolve().is_relative_to((ROOT/'videos').resolve());target.mkdir(parents=True,exist_ok=True);receipts=[];offset=np.array([0.,1.5,0.])
    renderer=mujoco.Renderer(model,height=720,width=1280)
    try:
        for index in selected:
            row=rows[index];name=row['clip'];length=row['full_clip_steps'];state=states[:length+1,index];ref=references[:length+1,index]
            actual=np.concatenate([state[:,:7],state[:,13:42]],1)
            desired=np.concatenate([ref[:,:3],Rotation.from_matrix(ref[:,3:12].reshape(-1,3,3)).as_quat(scalar_first=True),ref[:,12:]],1)
            out=target/f'{name}.mp4';assert a.verify_existing or not out.exists(),'Preserve previously published video';count=0
            if a.verify_existing:
                assert out.exists() and not out.with_suffix('.json').exists(),'Only finalize an unreceipted existing render'
                count=len(range(0,length+1,2))
            else:
              with imageio.get_writer(out,fps=25,codec='libx264',quality=8,macro_block_size=None,ffmpeg_params=['-pix_fmt','yuv420p']) as writer:
               for frame in range(0,length+1,2):
                    data.qpos[:36]=actual[frame];data.qpos[36:]=desired[frame];data.qpos[36:39]+=offset;mujoco.mj_kinematics(model,data)
                    distance=np.linalg.norm(actual[frame,:2]-desired[frame,:2]-offset[:2]);camera.distance=max(4.8,1.25*distance+2)
                    camera.lookat[:]=.5*(actual[frame,:3]+desired[frame,:3]+offset);camera.lookat[2]=.75
                    renderer.update_scene(data,camera=camera);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG]=False
                    picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture)
                    draw.rectangle((0,0,1280,82),fill='#122034');draw.text((18,10),f'G1 physical rollout | {name} | {frame*.02:.2f}s / {length*.02:.2f}s',fill='white',font=font)
                    draw.text((18,42),'Gray: actual physics. Green: reference +1.5 m Y display offset. All metrics use unshifted world.',fill='#c8d7e6',font=small)
                    draw.rectangle((0,628,1280,720),fill='#122034')
                    draw.text((18,637),f'XY error {errors[frame,index]:.3f} m | hard-failure samples {int(np.nansum(failed[:frame+1,index]))} | resets {row["post_initialization_resets"]}',fill='white',font=font)
                    gate='PASS' if row['passed_original_hard_gate'] else 'FAIL';quality_label=('PASS' if quality[name]['tracking_quality_passed'] else 'FAIL') if name in quality else 'not assessed in this overlay'
                    scope=('Full calibrated sequence' if summary.get('reference_version')=='continuous-corrected-v1' else 'Full raw frozen sequence') if continuous else ('Full calibrated clip' if summary.get('reference_variant')=='ground-calibration-v1' else 'Full raw clip')
                    if row['group']=='lafan':scope='Full unchanged LAFAN1 clip'
                    if summary.get('controller_variant')=='action-output-bound-20-v1':scope+=' | output bound [-20,20]'
                    draw.text((18,674),f'{scope}: original hard gate {gate} | tracking quality {quality_label} | failed tail retained',fill='#c5ddf0',font=font)
                    writer.append_data(np.asarray(picture));count+=1
            reader=imageio.get_reader(out);decoded=0;samples=[];positions=set(np.linspace(0,count-1,5).round().astype(int).tolist())
            try:
                for frame,picture in enumerate(reader):
                    assert picture.shape==(720,1280,3);decoded+=1
                    if frame in positions:samples.append(Image.fromarray(picture).resize((640,360)))
            finally:reader.close()
            assert decoded==count;sheet=Image.new('RGB',(1280,1080),'white')
            for i,picture in enumerate(samples):sheet.paste(picture,(i%2*640,i//2*360))
            sheet_path=ROOT/'evidence'/f'{video_label}-{name}-contact-sheet.jpg';sheet.save(sheet_path,quality=90)
            receipt={'clip':name,'file':str(out.relative_to(ROOT)).replace('\\','/'),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                     'frames':count,'fps':25,'video_seconds':count/25,'physical_window_seconds':length*.02,'complete_decode_passed':True,
                     'source_run':a.run,'only_retained_actual_physics_states_rendered':True,'reference_display_offset_m':[0,1.5,0],'fog_disabled_for_failed_tail_visibility':True,
                     'reference_version_or_variant':summary.get('reference_variant',summary.get('reference_version','raw-library')),
                     'original_hard_gate_passed':row['passed_original_hard_gate'],'tracking_quality_passed':quality.get(name,{}).get('tracking_quality_passed'),
                     'contact_sheet':str(sheet_path.relative_to(ROOT)).replace('\\','/')}
            (target/f'{name}.json').write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt);print('LIBRARY_VIDEO '+json.dumps(receipt),flush=True)
    finally:renderer.close()
    print('LIBRARY_RENDER_COMPLETE '+json.dumps({'run':a.run,'count':len(receipts),'complete_decode_passed':True}))


if __name__=='__main__':main()
