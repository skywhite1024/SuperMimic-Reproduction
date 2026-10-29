"""Audit all-40 evaluation contracts, checkpoint containers and every video frame."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile
import imageio_ffmpeg
from PIL import Image


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();root=args.results.resolve()
    assert read(root.parent/'local-transfer-verification.json')['status']=='archive_and_every_file_verified'
    old=read(root/'original-evaluation.json');names=list(old['per_clip'])
    assert len(names)==40
    index=read(root/'evaluations/index.json')
    assert index and len({Path(row['result']).name for row in index})==len(index)
    evaluations=[]
    for p in sorted((root/'evaluations').glob('*.json')):
        if p.name=='index.json':continue
        row=read(p);assert list(row['per_clip'])==names
        for name,clip in row['per_clip'].items():
            assert math.isfinite(clip['norm_return']) and math.isfinite(clip['reward_sum'])
            assert 0<clip['steps']<=clip['full_clip_steps']
            assert clip['full_clip_steps']==old['per_clip'][name]['full_clip_steps']
            assert abs(clip['normalization_denominator']-clip['full_clip_steps']*.1)<1e-9
            if not clip['failed']:assert clip['steps']==clip['full_clip_steps']
        assert abs(sum(c['norm_return'] for c in row['per_clip'].values())/40-row['norm_return'])<1e-5
        weighted=sum(c['reward_sum'] for c in row['per_clip'].values())/sum(c['normalization_denominator'] for c in row['per_clip'].values())
        assert abs(weighted-row['norm_return_framew'])<1e-5
        assert row['n_failed']==sum(bool(c['failed']) for c in row['per_clip'].values())
        assert set(row['failed'])=={name for name,c in row['per_clip'].items() if c['failed']}
        indexed=[item for item in index if Path(item['result']).name==p.name]
        assert len(indexed)==1 and indexed[0]['checkpoint_sha256']==row['checkpoint_sha256']
        checkpoint=root/'policy'/Path(indexed[0]['checkpoint']).name
        with checkpoint.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==row['checkpoint_sha256']
        evaluations.append({'file':p.name,'macro':row['norm_return'],'failed':row['n_failed'],'all40_full_denominators_passed':True})
    assert len(evaluations)==len(index)
    selection=read(root/'selection.json')
    primary=[row for row in index if not row['confirmation']]
    expected_best=min(primary,key=lambda row:(row['n_failed'],-row['macro'],-row['iteration']))
    assert selection['best']==expected_best
    assert selection['final']==max(primary,key=lambda row:row['iteration'])
    if selection['target_confirmed']:
        best=selection['best']
        assert best['macro']>=.82 and best['n_failed']==0
        assert any(row['confirmation'] and row['checkpoint_sha256']==best['checkpoint_sha256'] and row['macro']>=.82 and row['n_failed']==0 for row in index)
    identity=read(root/'evidence/loaded-identity.json');acceptance=read(root/'evidence/resume-acceptance.json')
    assert identity['passed'] and identity['full_optimizer_moments_checked']
    assert acceptance['passed'] and acceptance['before']['iteration']==12000 and acceptance['after']['iteration']==12005
    readback=read(root/'evidence/full-final-readback.json')
    assert readback['passed'] and readback['device']=='cpu' and not readback['GPU_operations_executed']
    assert readback['optimizer_moments_match_training_snapshot'] and readback['models_match_training_snapshot']
    with (root/'policy'/f'full_{selection["best"]["iteration"]:06d}.pt').open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==readback['full_checkpoint_sha256']
    checkpoints=[]
    for p in sorted((root/'policy').glob('*.pt')):
        with zipfile.ZipFile(p) as archive:
            assert archive.testzip() is None
        checkpoints.append({'file':p.name,'bytes':p.stat().st_size,'zip_crc_passed':True})
    frames=root.parent/'report/frames';frames.mkdir(parents=True,exist_ok=True)
    auxiliary=root/'postprocess-20261026-a'
    auxiliary_used=(auxiliary/'evidence/postprocess.json').is_file()
    trace_root=auxiliary if auxiliary_used else root
    if auxiliary_used:
        assert read(auxiliary/'evidence/postprocess.json')['status']=='complete'
        for name in ['parent-012000','selected']:
            summary=read(auxiliary/'diagnostics'/name/'summary.json')
            comparison=read(auxiliary/'diagnostics'/name/'primary-comparison.json')
            assert summary['independent_trace_contract_checked']
            assert len(comparison['per_clip'])==40
            assert summary['same_all40_rollout_metrics_reproduced']==comparison['strict_all40_match']
    videos=[]
    for p in sorted((trace_root/'videos').glob('*.mp4')):
        reader=imageio_ffmpeg.read_frames(str(p),pix_fmt='rgb24')
        meta=next(reader);width,height=meta['size'];count=0;samples=[];last=None
        for rgb in reader:
            assert len(rgb)==width*height*3
            if count in [0,50,250]:
                path=frames/f'{p.stem}-{count:04d}.png'
                Image.frombytes('RGB',(width,height),rgb).save(path)
                samples.append({'frame':count,'rgb_sha256':hashlib.sha256(rgb).hexdigest(),'path':str(path)})
            last=rgb;count+=1
        assert count>0 and last is not None
        Image.frombytes('RGB',(width,height),last).save(frames/(p.stem+'-last.png'))
        assert (width,height)==(960,540) and abs(meta['fps']-25)<1e-6
        videos.append({'file':p.name,'frames':count,'duration_seconds':count/meta['fps'],'size':[width,height],
                       'status':'all_frames_decoded','samples':samples})
    assert len(videos)==3
    result={'status':'downloaded_artifacts_audited','evaluations':evaluations,'checkpoints':checkpoints,
            'resume_actual_updates_verified':True,'full_optimizer_load_identity_verified':True,
            'final_full_checkpoint_CPU_deserialization_and_state_identity_verified':True,
            'videos':videos,'all_frames_decoded':sum(row['frames'] for row in videos),
            'auxiliary_independent_diagnostics_used':auxiliary_used,
            'local_model_inference_executed':False,'boundary':'Container CRC and byte hashes complement actual cloud model loading;evaluation remains the quality test. Auxiliary numerical replay differences remain separately recorded.'}
    (root.parent/'artifact-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'evaluations':len(evaluations),'checkpoints':len(checkpoints),'video_frames':result['all_frames_decoded']}))


if __name__=='__main__':main()
