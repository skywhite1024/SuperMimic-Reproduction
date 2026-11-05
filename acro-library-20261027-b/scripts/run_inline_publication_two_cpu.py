"""Separate finite GitHub publication phase; the former local repair window stays closed."""
import json,os,runpy,sys,time
from pathlib import Path
import psutil
root=Path(__file__).resolve().parents[1]
window=json.loads((root/'evidence/qualified-expert-v2-inline-publication-window.json').read_text(encoding='utf-8-sig'))
assert time.time()<window['deadline_epoch'],'Publication window expired; no automatic extension'
assert window['GPU']==0 and window['training_updates']==0 and window['SenseCore_changes']==0
process=psutil.Process();process.cpu_affinity(process.cpu_affinity()[:2]);assert len(process.cpu_affinity())<=2
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
target=Path(sys.argv[1]).resolve();assert target.is_relative_to(root)
sys.argv=sys.argv[1:];runpy.run_path(str(target),run_name='__main__')
