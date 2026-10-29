"""Decode every delivered video and save a few unmodified frames for visual QA.

Run with .asset-venv (imageio-ffmpeg 0.6.0, Pillow 11.3.0, NumPy).
Short videos remain valid evidence of early termination; they are not discarded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    root=args.results.resolve()
    frames_dir=root.parent/'report'/'frames'
    frames_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in sorted((root/'runs/videos').glob('*.mp4')):
        reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
        metadata=next(reader)
        width,height=metadata['size']
        fps=float(metadata['fps'])
        assert width==960 and height==540 and abs(fps-25)<0.01
        count=0
        samples=[]
        last=None
        try:
            for index,raw in enumerate(reader):
                assert len(raw)==width*height*3
                count+=1
                last=raw
                if index in (0,50,250):
                    array=np.frombuffer(raw,dtype=np.uint8).reshape(height,width,3)
                    output=frames_dir/f'{path.stem}-{index:04d}.png'
                    Image.fromarray(array).save(output)
                    samples.append({'frame':index,'time_seconds':index/fps,
                                    'rgb_sha256':hashlib.sha256(raw).hexdigest(),
                                    'pixel_std':float(array.std()),'path':str(output)})
        finally:
            reader.close()
        assert count>0 and last is not None
        assert count/fps<=15.1
        if count<=250:
            array=np.frombuffer(last,dtype=np.uint8).reshape(height,width,3)
            output=frames_dir/f'{path.stem}-last.png'
            Image.fromarray(array).save(output)
        records.append({'file':str(path),'status':'all_frames_decoded','frames':count,
                        'fps':fps,'duration_seconds':count/fps,'metadata':metadata,
                        'shorter_than_requested_15s':count/fps<14.9,'samples':samples,
                        'sampled_visual_motion':len({s['rgb_sha256'] for s in samples})>1})
    assert 2<=len(records)<=3
    audit={'status':'video_decode_passed','decoder_package':imageio_ffmpeg.__version__,
           'ffmpeg_version':imageio_ffmpeg.get_ffmpeg_version(),'videos':records,
           'boundary':'Decoding and sampled visual inspection do not replace tracking evaluation.'}
    (root.parent/'video-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':audit['status'],'videos':[{'frames':r['frames'],'seconds':r['duration_seconds']} for r in records]}))


if __name__=='__main__':
    main()
