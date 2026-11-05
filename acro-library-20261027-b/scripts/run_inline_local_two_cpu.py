"""Run this finite offline preparation on at most two local logical CPUs."""
import json,os,runpy,sys,time
from pathlib import Path
import psutil

root=Path(__file__).resolve().parents[1]
window=json.loads((root/'evidence/qualified-expert-v2-inline-local-window.json').read_text())
assert time.time()<window['deadline_epoch'],'The finite local preparation window has expired'
process=psutil.Process();process.cpu_affinity(process.cpu_affinity()[:2]);assert len(process.cpu_affinity())<=2
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
target=Path(sys.argv[1]).resolve();assert target.is_relative_to(root)
sys.argv=sys.argv[1:];runpy.run_path(str(target),run_name='__main__')
