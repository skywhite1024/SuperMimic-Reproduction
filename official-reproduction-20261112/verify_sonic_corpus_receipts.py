"""Recover SONIC originals and independently recompute all40 native clip scores."""
import argparse
import gzip
import json
import math
from pathlib import Path
import re

import numpy as np

from verify_horizon_grid_receipts import decode_group,sha


FILES=["inference-wheel-verification.json","setup.json","models-batched/validation.json","evaluation/protocol.json",
       "evaluation/result.json","evaluation/accounting.json","evaluation/audit/summary.json",
       "evaluation/audit/initial-states.json","result.json"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["log","plan","motions","output"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--require-complete",action="store_true")
    args=p.parse_args();plan=json.loads(args.plan.read_bytes())
    raw=args.log.read_bytes()
    if args.log.suffix==".gz":raw=gzip.decompress(raw)
    clean="\n".join(re.sub(r"^\[[0-9TZ: .+-]+\] ","",line) for line in raw.decode().splitlines())
    assert "SONIC_BOOTSTRAP_SHA256 "+plan["command_sha256"] in clean
    assert all("SONIC_SOURCE_SHA256 "+name+" "+value in clean for name,value in plan["source_files"].items())
    args.output.mkdir(parents=True,exist_ok=True)
    blocks=re.findall(r"^SONIC_ORIGINALS_BEGIN\n(.*?)^SONIC_ORIGINALS_END$",clean,re.M|re.S)
    assert len(blocks)<=1
    validation=dict(status="originals_not_yet_emitted",original_JSONs=0,paper_results_reproduced=False)
    ledger=[]
    if blocks:
        names,ledger=decode_group(blocks[0],plan["output"],args.output,FILES)
        read=lambda name:json.loads((args.output/name).read_bytes())
        digest=lambda name:sha((args.output/name).read_bytes())
        report=read("result.json")
        validation.update(status="failed_or_incomplete_originals_retained",original_JSONs=len(names),
                          reported_status=report["status"],error=report.get("error"))
        if report["status"]=="complete":
            assert names==FILES and "SONIC_FINITE_JOB_COMPLETED" in clean and "SONIC_JOB_EXIT=0" in clean
            assert all(report["originals"][name]==digest(name) for name in FILES[:-1])
            setup=read("setup.json");protocol=read("evaluation/protocol.json")
            wheels=read("inference-wheel-verification.json")
            assert digest("inference-wheel-verification.json")==plan["source_files"]["sonic-linux-inference-wheels.json"]
            assert len(wheels["wheels"])==2 and all(len(row["sha256"])==64 for row in wheels["wheels"])
            result=read("evaluation/result.json");account=read("evaluation/accounting.json")
            arrays=read("evaluation/audit/summary.json");states=read("evaluation/audit/initial-states.json")
            local_files={p.name:sha(p.read_bytes()) for p in args.motions.iterdir() if p.is_file()}
            assert setup["common_corpus_files"]==protocol["motion_files"]==local_files
            motion_names=json.loads((args.motions/"motion_names.json").read_bytes())
            ends=np.load(args.motions/"motion_ends.npy");lengths=np.diff(np.r_[0,ends])
            assert len(motion_names)==len(set(motion_names))==len(ends)==40 and ends[-1]==441140
            assert setup["HF_revision"]==plan["HF_revision"]
            assert protocol["SONIC_ONNX_provider"]==result["ONNX_provider"]=="CPUExecutionProvider"
            assert protocol["all_compiled_CPU_and_live_CUDA_control_fields_verified"]
            assert protocol["SONIC_control_contract_sha256"]==plan["source_files"]["sonic-g1-control-contract.json"]
            assert len(protocol["compiled_SONIC_control_readback"])==29
            assert result["status"]=="full_corpus_evaluation_completed"
            assert result["clips"]==account["clips"]==40 and len(result["records"])==40
            assert sorted(r["index"] for r in result["records"])==list(range(40))
            assert result["initial_reference_state_verified_for_all_clips"] and protocol["manual_reset_version"]==2
            total=0.;successes=steps=0
            for row in result["records"]:
                i=row["index"];n=int(lengths[i])
                assert row["name"]==motion_names[i] and row["reference_frames"]==n
                assert isinstance(row["steps"],int) and 0<row["steps"]<=n and math.isfinite(row["reward_sum"])
                native_success="motion_end" in row["termination_terms"] and not any(
                    not protocol["termination_terms"][name]["time_out"] for name in row["termination_terms"])
                assert row["success"]==native_success and (not native_success or row["steps"]==n)
                assert row["reward_sum"]<=row["steps"]*.1+1e-4
                assert math.isclose(row["seconds"],row["steps"]*.02,abs_tol=1e-6)
                total+=row["reward_sum"]/(n*.1);successes+=int(native_success);steps+=row["steps"]
            assert successes==result["successes"]==account["successes"]==arrays["verified_successes"]
            assert successes+result["failures"]==40
            assert math.isclose(total/40,account["mean_normalized_full_motion_return"],rel_tol=1e-13)
            assert account["maximum_step_reward"]==.1
            assert account["result_sha256"]==arrays["result_sha256"]==digest("evaluation/result.json")
            assert account["protocol_sha256"]==states["protocol_sha256"]==digest("evaluation/protocol.json")
            assert arrays["accounting_sha256"]==digest("evaluation/accounting.json")
            assert states["trajectory_integrity_summary_sha256"]==digest("evaluation/audit/summary.json")
            assert arrays["all_trajectory_arrays_verified"] and arrays["verified_steps"]==steps and arrays["verified_clips"]==40
            assert states["all_saved_initial_states_verified"] and states["saved_initial_states_verified"]==40
            validation.update(status="pretrained_SONIC_full40_originals_and_native_score_recomputation_verified",
                clips=40,successes=successes,failures=40-successes,mean_normalized_full_motion_return=total/40,
                all_cloud_arrays_and_initial_states_audited=True,ONNX_GPU_provider_used=False,
                author_exact_filtered129514_benchmark=False,SONIC_training_reset_confirmed=False,
                all_native_reward_termination_records_verified=True)
    validation.update(job_uid=plan["uid"],source_commit=plan["source_commit"],log_sha256=sha(raw),
                      raw_trajectory_arrays_downloaded=False,original_bytes_verified=bool(ledger))
    for name,value in [("validation.json",validation),("receipt-ledger.json",dict(receipts=ledger))]:
        (args.output/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8",newline="\n")
    if args.require_complete:assert validation["status"]=="pretrained_SONIC_full40_originals_and_native_score_recomputation_verified"
    print(json.dumps(validation,indent=2))


if __name__=="__main__":main()
