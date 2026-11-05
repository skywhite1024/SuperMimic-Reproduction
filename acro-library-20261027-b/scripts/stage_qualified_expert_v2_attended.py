"""One finite, strictly pinned upload/preflight; the operator closes CCI in the UI."""
import hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LABEL='v2stage1030c'
CLOUD='/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b'
HOST='root@203.0.113.1'

def main():
    window=json.loads((ROOT/f'evidence/cci-{LABEL}-retrieval-window.json').read_text())
    assert time.time()<window['deadline_epoch']-420,'Reserve seven minutes for preflight and verified UI closure'
    known=ROOT/f'evidence/cci-{LABEL}-known-hosts'; assert known.is_file()
    key=Path.home()/'.ssh/sensecore_cci_ed25519';assert key.is_file()
    options=['-i',str(key),'-o','StrictHostKeyChecking=yes','-o',f'UserKnownHostsFile={known}',
             '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10']
    ssh=['ssh','-p','10076',*options,HOST];scp=['scp','-P','10076',*options]
    log=ROOT/f'evidence/cci-{LABEL}-staging-console.txt';assert not log.exists()
    receipt={'started_at':datetime.now(timezone.utc).isoformat(),'label':LABEL,'GPU':0,'training_updates':0,
             'strict_host_key_checking':True,'private_key_body_read':False,'manual_resource_closure_required':True}
    def run(command,data=None,timeout=180):
        assert time.time()<window['deadline_epoch']-240,'Begin resource closure now'
        result=subprocess.run(command,input=data,text=True,encoding='utf8',errors='replace',stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,timeout=min(timeout,max(1,int(window['deadline_epoch']-time.time()-240))))
        with log.open('a',encoding='utf8') as stream:stream.write(result.stdout+'\n')
        print(result.stdout[-5000:],flush=True)
        assert result.returncode==0,f'Finite staging step failed: {result.returncode}'
        return result.stdout
    incoming=f'{CLOUD}/deployment/staging-{LABEL}'
    try:
        run([*ssh,'python3 -'],"from pathlib import Path\nimport os\nos.umask(0o077)\nassert Path('/sys/fs/cgroup/cpu.max').read_text().strip()=='200000 100000'\nassert int(Path('/sys/fs/cgroup/memory.max').read_text())==8589934592\np=Path("+repr(incoming)+")\nassert p.resolve().is_relative_to(Path('/mnt/afs/USER_ID'))\np.mkdir(mode=0o700)\nprint('STRICT_SSH_OWN_AFS_2CPU_8GIB_PASS')\n",timeout=30)
        files=[ROOT/'deployment/qualified-expert-v2-source-and-teacher.tar.gz',ROOT/'deployment/qualified-expert-v2-control-supplement.tar.gz',ROOT/'scripts/stage_qualified_expert_v2.py']
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        assert hashes[files[0].name]=='3edc27b42a47a16adfb5ab1c8f1e95ec8ec754a345ac07e2ec17b2fe0f2d031f'
        assert hashes[files[1].name]=='0b809a4011fbc6b66fe197e776d59104be545cfc08d45362e8145d516cc14712'
        assert hashes[files[2].name]=='75c4780c9ec82f50e6fe24907f7a434c453866944465147406a24131abc76985'
        run([*scp,*map(str,files),f'{HOST}:{incoming}/'],timeout=90)
        check="from pathlib import Path\nimport hashlib\np=Path("+repr(incoming)+")\nexpected="+repr(hashes)+"\nfor name,digest in expected.items():\n assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name\nprint('ALL_UPLOADED_BYTES_SHA256_PASS')\n"
        run([*ssh,'python3 -'],check,timeout=30)
        venv='/mnt/afs/USER_ID/SuperMimic/lafan1-20261024/uni/.venv/bin/python'
        run([*ssh,f'timeout --signal=TERM --kill-after=10s 150s {venv} -u {incoming}/stage_qualified_expert_v2.py --archive {incoming}/{files[0].name} --control-archive {incoming}/{files[1].name} --root {CLOUD}'],timeout=170)
        target=ROOT/'evidence/qualified-expert-v2-cloud-CPU-preflight.json';assert not target.exists()
        run([*scp,f'{HOST}:{CLOUD}/evidence/qualified-expert-v2-cloud-CPU-preflight.json',str(target)],timeout=30)
        proof=json.loads(target.read_text());assert proof['new_training_updates']==0 and proof['CUDA_visible_devices']==0
        receipt.update(status='CPU_only_staging_and_preflight_passed_pending_resource_closure',preflight_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),uploaded_sha256=hashes)
    except BaseException as error:
        receipt.update(status='staging_failed_preserved_pending_resource_closure',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt['ended_at']=datetime.now(timezone.utc).isoformat()
        (ROOT/f'evidence/cci-{LABEL}-staging-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
