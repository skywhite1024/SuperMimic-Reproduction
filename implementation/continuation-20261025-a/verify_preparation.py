"""Local checks of scheduling, payload integrity and real archive round trips."""
from pathlib import Path
from types import SimpleNamespace
import ast
import base64
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from audit_common import lr_at

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    count=0
    for p in ROOT.glob('*.py'):
        compile(p.read_text(encoding='utf-8'),str(p),'exec');count+=1
    learner=BASE/'uni/src/unisupermimic/rl/algos/supermimic/learner.py'
    tree=ast.parse(learner.read_text())
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='lr_at')
    namespace={'math':math}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(learner),'exec'),namespace)
    cfg=SimpleNamespace(cfg=SimpleNamespace(lr=2e-4,lr_decay_iters=250000))
    for iteration in [0,11999,12000,12005,16000,20000,24000,250000,320000]:
        assert abs(namespace['lr_at'](cfg,iteration)-lr_at(iteration))<1e-15
    frozen=json.loads((ROOT/'frozen.json').read_text())
    for name,expected in frozen['base_files'].items():
        p=BASE/'downloaded'/name
        if not p.is_file():p=BASE/name
        assert sha(p)==expected,name
    cpu=(ROOT/'cpu-preflight.command.txt').read_text()
    bootstrap=cpu.split("python3 - <<'PY'\n",1)[1].split('\nPY\n',1)[0]
    bootstrap_tree=ast.parse(bootstrap)
    encoded=next(n.args[0].value for n in ast.walk(bootstrap_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    payload=json.loads(gzip.decompress(base64.b64decode(encoded)))
    assert len(payload)==7
    for name,body in payload.items(): assert body.encode()==(ROOT/name).read_bytes(),name
    for name in ['cpu-preflight.command.txt','gpu-continuation.command.txt']:
        subprocess.run([r'[local]/Program Files\Git\bin\bash.exe','-n',str(ROOT/name)],check=True)
    # Exercise the real collector and verifier with a tiny interrupted-run fixture.
    with tempfile.TemporaryDirectory(prefix='continuation-check-',dir=ROOT) as folder:
        tmp=Path(folder);fixture=tmp/'experiment';fixture.mkdir();fixture_base=tmp/'base'
        old=fixture_base/'runs/baseline-evaluation.json';old.parent.mkdir(parents=True)
        old.write_text('{"fixture":true}')
        for name in ['audit_common.py','collect_continuation.py']:shutil.copyfile(ROOT/name,fixture/name)
        (fixture/'experiment.json').write_text('{"status":"failed_or_interrupted"}')
        (fixture/'frozen.json').write_text(json.dumps({'base_files':{'runs/baseline-evaluation.json':sha(old)}}))
        env={**os.environ,'SUPERMIMIC_BASE':str(fixture_base)}
        subprocess.run([sys.executable,str(fixture/'collect_continuation.py')],env=env,capture_output=True,check=True)
        subprocess.run([sys.executable,str(BASE/'verify_results.py'),'--archive',str(fixture/'continuation-results.tar.gz'),
                        '--receipt',str(fixture/'result-transfer-receipt.json'),'--output',str(tmp/'unpacked')],capture_output=True,check=True)
    result={'status':'local_preparation_verified','python_files_compiled':count,'learning_rate_matches_upstream_absolute_schedule_at_nine_points':True,
            'frozen_base_hashes_verified':len(frozen['base_files']),'deployed_payload_bytes_exact':True,'bash_syntax_passed':2,
            'collector_and_verifier_real_round_trip':'passed on a tiny interrupted-run fixture',
            'gpu_or_cloud_execution_in_this_check':False,'boundary':'Actual cloud parent existence,load identity,optimizer updates and GPU behavior still require finite live acceptance.'}
    (ROOT/'local-preparation-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
