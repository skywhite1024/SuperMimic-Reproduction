"""Pack all18 canonical walk/run/sprint clips without changing their physical rows."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    spec=json.loads(Path(__file__).with_name('lafan-locomotion18-spec.json').read_bytes())
    source=args.source.resolve();output=args.output.resolve()
    for name,value in spec['source_motion_files'].items():assert digest(source/name)==value
    names=json.loads((source/'motion_names.json').read_bytes())
    ends=np.load(source/'motion_ends.npy');starts=np.r_[0,ends[:-1]]
    assert len(names)==len(ends)==40 and int(ends[-1])==441140
    rows=spec['selected'];assert len(rows)==18
    assert [r['index'] for r in rows]==[i for i,name in enumerate(names) if name.startswith(('walk','run','sprint'))]
    for row in rows:
        i=row['index'];assert names[i]==row['name'] and int(ends[i]-starts[i])==row['reference_frames']
    output.mkdir(parents=True,exist_ok=False)
    (output/'joint_names.json').write_bytes((source/'joint_names.json').read_bytes())
    (output/'motion_names.json').write_text(json.dumps([r['name'] for r in rows],indent=2)+'\n',encoding='utf-8',newline='\n')
    np.save(output/'motion_ends.npy',np.cumsum([r['reference_frames'] for r in rows],dtype=ends.dtype))
    for name in spec['source_motion_files']:
        if not name.endswith('.npy') or name=='motion_ends.npy':continue
        original=np.load(source/name,mmap_mode='r')
        packed=np.concatenate([original[starts[r['index']]:ends[r['index']]] for r in rows])
        assert packed.shape[0]==spec['reference_frames'] and np.isfinite(packed).all()
        np.save(output/name,packed)
        saved=np.load(output/name,mmap_mode='r');cursor=0
        for row in rows:
            size=row['reference_frames'];i=row['index']
            assert np.array_equal(saved[cursor:cursor+size],original[starts[i]:ends[i]])
            cursor+=size
    receipt=dict(status='all18_original_motion_slices_preserved',source_motion_files=spec['source_motion_files'],
        selected=rows,clips=18,reference_frames=spec['reference_frames'],
        packed_motion_files={p.name:digest(p) for p in sorted(output.iterdir())},
        physical_rows_and_dtypes_unchanged=True,author_exact_Figure5_membership_confirmed=False)
    (output.parent/'motion-subset.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('LAFAN_LOCOMOTION18_PACKED '+json.dumps({key:receipt[key] for key in ('status','clips','reference_frames')}),flush=True)


if __name__=='__main__':main()
