"""Decode every frame of a generated simulation video and retain samples for visual review."""
import argparse,hashlib,json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from runtime import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
    path=ROOT/'videos'/a.name;reader=imageio.get_reader(path);meta=reader.get_meta_data();samples={}
    count=0
    for count,frame in enumerate(reader,1):
        assert frame.shape==(720,1280,3)
        if count-1 in [0,750,1500,2250,2999]:
            picture=Image.fromarray(frame);picture.save(ROOT/'evidence'/f'{path.stem}-decoded-{count-1:04d}.jpg',quality=90);samples[count-1]=picture
    reader.close();assert count==3000 and abs(meta['fps']-25)<1e-6 and abs(meta['duration']-120)<.05
    sheet=Image.new('RGB',(1280,3*390),'#eef3f8');d=ImageDraw.Draw(sheet)
    for i,(idx,picture) in enumerate(samples.items()):
        picture.thumbnail((630,355));x=(i%2)*640;y=(i//2)*390;sheet.paste(picture,(x,y));d.text((x+8,y+360),f'{idx/25:.2f} seconds',fill='#19354e')
    sheet.save(ROOT/'evidence'/f'{path.stem}-contact-sheet.jpg',quality=92)
    proof={'video':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'frames_decoded':count,'fps':meta['fps'],'duration_seconds':meta['duration'],'resolution':[1280,720],
           'complete_decode_passed':True,'sample_frames':list(samples),'visual_review':'pending'}
    (ROOT/'evidence'/f'video-{path.stem}.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))


if __name__=='__main__':main()
