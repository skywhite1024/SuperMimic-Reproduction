"""Read completed immutable trajectories and export finite, explicit pose samples."""
import argparse
import base64
import csv
import hashlib
import io
import json
import re
from pathlib import Path
import subprocess
import sys

import numpy as np


def digest(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--case",action="append",choices=["ppo-wm-4k32-1115i","ppo-wm-8k24-1115j",
        "ppo-wm-8k32-1115k","ppo-wm-16k32-1115l"],help="Explicit completed cases; default retains the original three-case readback")
    args=p.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    results=[]
    setup_log=args.root/"runs/sonic-lafan40-1115n/install-inference-dependencies.log"
    if setup_log.is_file():
        text=setup_log.read_text(encoding="utf-8",errors="replace")
        text=re.sub(r"(https?://)[^/\s@]+@",r"\1[redacted]@",text)
        text=re.sub(r"(?i)((?:token|password|api_key)=)[^\s&]+",r"\1[redacted]",text)
        diagnostic=dict(status="retained_SONIC_install_log_tail_readonly",source_sha256=digest(setup_log),
            source_bytes=setup_log.stat().st_size,tail="\n".join(text.splitlines()[-100:]),
            URL_userinfo_and_credential_query_values_redacted=True,environment_variables_not_read=True)
        path=args.output/"sonic-install-diagnostic.json"
        path.write_text(json.dumps(diagnostic,indent=2)+"\n",encoding="utf-8",newline="\n")
        print("SONIC_SETUP_DIAGNOSTIC_BEGIN",flush=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name("emit_json_receipts.py")),str(path)],check=True,timeout=120)
        print("SONIC_SETUP_DIAGNOSTIC_END",flush=True)
    cases=args.case or ["ppo-wm-4k32-1115i","ppo-wm-8k24-1115j","ppo-wm-8k32-1115k"]
    assert len(cases)==len(set(cases))
    for case in cases:
        parent=args.root/"runs"/case;folder=parent/"eval-002048";motions=parent/"motions"
        report=json.loads((parent/"result.json").read_bytes())
        assert report["status"]=="complete"
        result=json.loads((folder/"result.json").read_bytes())
        summary=json.loads((folder/"audit/summary.json").read_bytes())
        assert digest(folder/"result.json")==summary["result_sha256"]
        ledger=folder/"audit"/summary["ledger"]["file"]
        assert digest(ledger)==summary["ledger"]["sha256"]
        rows={r["trajectory"]:r for r in csv.DictReader(ledger.open(encoding="utf-8",newline=""))}
        ends=np.load(motions/"motion_ends.npy");starts=np.r_[0,ends[:-1]]
        sources={name:np.load(motions/(name+".npy"),mmap_mode="r") for name in
            ["root_pos_w","root_quat_w","joint_pos","root_lin_vel_w","root_ang_vel_w","joint_vel"]}
        for name,value in json.loads((folder/"protocol.json").read_bytes())["motion_files"].items():
            assert digest(motions/name)==value
        score=lambda row:row["reward_sum"]/(row["reference_frames"]*.1)
        successes=sorted([r for r in result["records"] if r["success"]],key=score)
        failures=sorted([r for r in result["records"] if not r["success"]],key=lambda r:r["index"])
        selected=[r["index"] for r in [*successes[:1],*successes[-1:],*failures[:1]]]
        arrays={};metrics=[]
        for row in result["records"]:
            path=folder/row["trajectory"];raw=path.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==rows[path.name]["sha256"]
            with np.load(io.BytesIO(raw),allow_pickle=False) as saved:
                state=saved["full_state"];actions=saved["actions"];rewards=saved["rewards"]
            n=row["steps"];index=row["index"]
            assert state.shape==(n,71) and actions.shape==(n,29) and rewards.shape==(n,)
            assert all(np.isfinite(a).all() for a in [state,actions,rewards])
            assert np.isclose(rewards.sum(dtype=np.float64),row["reward_sum"],atol=1e-9,rtol=0)
            frame=int(starts[index])+np.arange(n)
            reference=np.concatenate([value[frame] for value in sources.values()],axis=1)
            assert reference.shape==state.shape
            root=np.linalg.norm(state[:,:3]-reference[:,:3],axis=1)
            joints=np.sqrt(np.mean((state[:,7:36]-reference[:,7:36])**2,axis=1))
            item=dict(index=index,name=row["name"],native_success=row["success"],steps=n,
                native_normalized_full_motion_return=score(row),mean_world_root_distance=float(root.mean()),
                maximum_world_root_distance=float(root.max()),final_world_root_distance=float(root[-1]),
                mean_joint_RMS_degrees=float(np.rad2deg(joints.mean())),
                maximum_joint_RMS_degrees=float(np.rad2deg(joints.max())),
                full_trajectory_sha256=rows[path.name]["sha256"],selected_for_pose_visualization=index in selected)
            if index in selected:
                ids=np.unique(np.r_[np.arange(min(n,300)),
                    np.arange(max(0,n//2-150),min(n,n//2+150)),np.arange(max(0,n-300),n)]).astype(np.int64)
                prefix=str(index)+"_"
                for name,value in [("frame",ids),("state",state[ids]),("reference",reference[ids]),
                    ("actions",actions[ids]),("reward",rewards[ids]),("root_error",root[ids]),("joint_rms",joints[ids])]:
                    arrays[prefix+name]=value
                item["sampled_frames"]=len(ids)
            metrics.append(item)
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);payload=buffer.getvalue()
        exported=dict(status="all18_final_trajectory_files_rehashed_and_all_arrays_read",
            case=case,source_result_sha256=digest(folder/"result.json"),
            source_audit_sha256=digest(folder/"audit/summary.json"),source_ledger_sha256=digest(ledger),
            native_joint_names=json.loads((motions/"joint_names.json").read_bytes()),
            all18_metrics=metrics,selected_indices=selected,
            selection="Lowest-return native success, highest-return native success, first indexed failure; retain all18 numeric metrics",
            sample_rule="First300, middle300 and last300 saved pre-action frames, deduplicated; not contiguous across windows",
            npz_base64=base64.b64encode(payload).decode(),npz_bytes=len(payload),
            npz_sha256=hashlib.sha256(payload).hexdigest(),
            simulation_reexecuted=False,paper_results_reproduced=False)
        target=args.output/(case+".json")
        data=(json.dumps(exported,separators=(",",":"))+"\n").encode()
        assert len(data)<4_000_000,(case,len(data))
        target.write_bytes(data);results.append(target)
        print("PPO_WM_QUALITY_READBACK "+json.dumps(dict(case=case,all_clips=18,selected=selected,npz_bytes=len(payload))),flush=True)
    print("PPO_WM_QUALITY_ORIGINALS_BEGIN",flush=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name("emit_json_receipts.py")),
                    "--max-bytes","4000000",*[str(p) for p in results]],check=True,timeout=120)
    print("PPO_WM_QUALITY_ORIGINALS_END",flush=True)


if __name__=="__main__":main()
