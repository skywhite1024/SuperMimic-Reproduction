"""Finite pretrained SONIC comparison on the existing canonical LAFAN40 pack."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

from prepare_sonic_onnx_batch import MODEL_HASHES,digest


def write(path,data):
    path.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8",newline="\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve();output=args.output.resolve()
    assert output.is_relative_to(root/"runs") and not output.exists()
    output.mkdir(parents=True)
    source=Path(__file__).resolve().parent
    inference=root/"conda/sonic-inference-cpu"
    executable=inference/"bin/python"
    models=root/"data/sonic-default-6733128"
    motions=root/"data/lafan40-gpu-1112b/packed"
    contract=source/"sonic-g1-control-contract.json"
    stages=[];error=None
    started=time.monotonic()
    def stage(name,command,limit):
        log=output/(name+".log")
        print("SONIC_STAGE_START "+json.dumps(dict(stage=name,utc=datetime.now(timezone.utc).isoformat(),limit_seconds=limit)),flush=True)
        beginning=time.monotonic()
        timed_out=False
        with log.open("w",encoding="utf-8") as stream:
            try:
                process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=limit)
                returncode=process.returncode
            except subprocess.TimeoutExpired:
                timed_out=True;returncode=124
        row=dict(stage=name,returncode=returncode,timed_out=timed_out,
                 elapsed_seconds=time.monotonic()-beginning,log_sha256=digest(log))
        stages.append(row)
        for line in log.read_text(encoding="utf-8",errors="replace").splitlines():
            if any(tag in line for tag in ["SONIC_","CORPUS_CLIP_RESULT","CORPUS_PROGRESS","OFFICIAL_CORPUS_EVAL_PASS","CORPUS_ACCOUNTING","AUDIT_"]):
                print(line,flush=True)
        print("SONIC_STAGE_EXIT "+json.dumps(row),flush=True)
        if returncode:
            print(log.read_text(encoding="utf-8",errors="replace")[-14000:],flush=True)
            raise RuntimeError(name+" failed with exit "+str(returncode))
    try:
        if not executable.exists():
            stage("create-inference-venv",[sys.executable,"-m","venv","--system-site-packages",str(inference)],180)
        assert sys.version_info[:2]==(3,13),"Pinned Linux wheel manifest targets the established CPython3.13 runtime"
        wheel_manifest=json.loads((source/"sonic-linux-inference-wheels.json").read_bytes())
        requirements=output/"inference-requirements.txt"
        requirements.write_text("\n".join(row["package"]+"=="+row["version"]+
            " --hash=sha256:"+row["sha256"] for row in wheel_manifest["wheels"])+"\n",encoding="utf-8")
        wheel_cache=root/"data/sonic-inference-wheels";wheel_cache.mkdir(parents=True,exist_ok=True)
        cached=all((wheel_cache/row["file"]).is_file() and digest(wheel_cache/row["file"])==row["sha256"]
                   for row in wheel_manifest["wheels"])
        if not cached:
            stage("download-pinned-inference-wheels",[str(executable),"-m","pip","download",
                "--disable-pip-version-check","--no-deps","--only-binary=:all:","--require-hashes",
                "--timeout","30","--retries","1","--index-url",wheel_manifest["mirror_index"],
                "--dest",str(wheel_cache),"-r",str(requirements)],300)
        for row in wheel_manifest["wheels"]:
            path=wheel_cache/row["file"]
            assert path.stat().st_size==row["bytes"] and digest(path)==row["sha256"]
        write(output/"inference-wheel-verification.json",wheel_manifest)
        stage("install-inference-dependencies-offline",[str(executable),"-m","pip","install",
            "--disable-pip-version-check","--no-index","--no-deps","--require-hashes",
            "--find-links",str(wheel_cache),"-r",str(requirements)],300)
        expected_files=dict(MODEL_HASHES)
        expected_files.update({"observation_config.yaml":"466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b",
            "sonic_release/config.yaml":"f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7"})
        download_code = (
            "import hashlib,json,sys,urllib.request,urllib.error,uuid\n"
            "from pathlib import Path\n"
            "root=Path(sys.argv[1]);expected=json.loads(sys.argv[2]);root.mkdir(parents=True,exist_ok=True)\n"
            "def digest(p):\n"
            " with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()\n"
            "for name,sha in expected.items():\n"
            " target=root/name;target.parent.mkdir(parents=True,exist_ok=True)\n"
            " if target.exists():\n"
            "  assert digest(target)==sha,('Existing model cache checksum mismatch',name)\n"
            "  print('SONIC_MODEL_TRANSFER '+json.dumps(dict(file=name,status='verified_cache')),flush=True);continue\n"
            " completed=False\n"
            " for endpoint in ['https://huggingface.co','https://hf-mirror.com']:\n"
            "  temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.part')\n"
            "  url=endpoint+'/nvidia/GEAR-SONIC/resolve/6733128a3d8a523b1418b06bca3cdf61c8b0987f/'+name\n"
            "  print('SONIC_MODEL_TRANSFER '+json.dumps(dict(file=name,endpoint=endpoint,status='starting')),flush=True)\n"
            "  try:\n"
            "   request=urllib.request.Request(url,headers={'User-Agent':'SuperMimic-Reproduction/1.0'})\n"
            "   with urllib.request.urlopen(request,timeout=60) as response,temporary.open('xb') as stream:\n"
            "    count=0\n"
            "    while chunk:=response.read(1024*1024):\n"
            "     stream.write(chunk);count+=len(chunk)\n"
            "     assert count<=80000000,('Unexpected model-file size',name,count)\n"
            "   assert digest(temporary)==sha,('Downloaded model checksum mismatch',name)\n"
            "   temporary.replace(target);completed=True\n"
            "   print('SONIC_MODEL_TRANSFER '+json.dumps(dict(file=name,endpoint=endpoint,status='SHA256_verified',bytes=count)),flush=True);break\n"
            "  except (urllib.error.URLError,TimeoutError) as error:\n"
            "   print('SONIC_MODEL_TRANSFER '+json.dumps(dict(file=name,endpoint=endpoint,status='transport_failed',error_type=type(error).__name__)),flush=True)\n"
            "   if isinstance(error,urllib.error.HTTPError) and error.code in [401,403]:raise\n"
            " if not completed:raise RuntimeError('Public pinned model transfer failed: '+name)\n")
        stage("download-public-pinned-models",[str(executable),"-c",download_code,str(models),json.dumps(expected_files)],900)
        assert all(digest(models/name)==value for name,value in MODEL_HASHES.items())
        assert digest(models/"observation_config.yaml")=="466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b"
        assert digest(models/"sonic_release/config.yaml")=="f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7"
        setup=dict(status="pinned_original_SONIC_model_SHA256_verified",
            original_models=MODEL_HASHES,HF_revision="6733128a3d8a523b1418b06bca3cdf61c8b0987f",
            source_commit="1e1ec4f5f27e6c3e03048f693bd7c60cce04afda",
            common_corpus_files={p.name:digest(p) for p in sorted(motions.iterdir()) if p.is_file()},
            control_contract_sha256=digest(contract),
            inference_provider="CPUExecutionProvider",physics_provider="CUDA MuJoCo Warp",
            model_training_executed=False,paper_results_reproduced=False)
        assert setup["common_corpus_files"]==json.loads((source/"lafan-locomotion18-spec.json").read_bytes())["source_motion_files"]
        write(output/"setup.json",setup)
        batch=output/"models-batched"
        stage("derive-and-check-batched-models",[str(executable),str(source/"prepare_sonic_onnx_batch.py"),
            "--models",str(models),"--output",str(batch)],300)
        evaluation=output/"evaluation"
        stage("evaluate-full40",[str(executable),str(source/"evaluate_sonic_corpus_v2.py"),
            "--models",str(models),"--batched",str(batch),"--contract",str(contract),
            "--motions",str(motions),"--output",str(evaluation),"--expected-clips","40","--num-envs","40"],1200)
        stage("account-full40",[str(executable),str(source/"audit_corpus_result.py"),
            "--result",str(evaluation/"result.json"),"--protocol",str(evaluation/"protocol.json"),
            "--motions",str(motions),"--expected-clips","40","--require-trajectories",
            "--output",str(evaluation/"accounting.json")],180)
        stage("audit-all-arrays-and-initial-states",[str(executable),str(source/"audit_saved_initial_states_v2.py"),
            "--evaluation",str(evaluation),"--motions",str(motions),"--expected-clips","40",
            "--output",str(evaluation/"audit")],180)
    except Exception as exception:
        error=type(exception).__name__+": "+str(exception)
        print("SONIC_FINITE_JOB_ERROR "+error,flush=True)
    files=["inference-wheel-verification.json","setup.json","models-batched/validation.json","evaluation/protocol.json",
        "evaluation/result.json","evaluation/accounting.json","evaluation/audit/summary.json",
        "evaluation/audit/initial-states.json"]
    complete=not error and all((output/name).is_file() for name in files)
    result=dict(status="complete" if complete else "failed_or_incomplete",error=error,
        elapsed_seconds=time.monotonic()-started,stages=stages,
        originals={name:digest(output/name) for name in files if (output/name).is_file()},
        benchmark="All40 common canonical LAFAN clips, corrected-v2 native reward/termination; not author filtered129514 benchmark",
        full40_completed=complete,all_cloud_arrays_and_initial_states_audited=complete,
        GPU_physics_executed=any(s["stage"]=="evaluate-full40" and s["returncode"]==0 for s in stages),
        ONNX_GPU_provider_used=False,SONIC_training_reset_confirmed=False,paper_results_reproduced=False)
    write(output/"result.json",result)
    existing=[output/name for name in files if (output/name).is_file()]+[output/"result.json"]
    print("SONIC_ORIGINALS_BEGIN",flush=True)
    subprocess.run([sys.executable,str(source/"emit_json_receipts.py"),*[str(p) for p in existing]],check=True,timeout=120)
    print("SONIC_ORIGINALS_END",flush=True)
    print("SONIC_FINITE_JOB_COMPLETED" if complete else "SONIC_FINITE_JOB_FAILED",flush=True)
    if not complete:raise RuntimeError(error or "Missing originals")


if __name__=="__main__":main()
