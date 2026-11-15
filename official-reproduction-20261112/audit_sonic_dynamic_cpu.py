"""Read every saved scalar pilot array; verify common-adapter input/action parity."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_sonic_onnx_batch import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["pilots","motions","models","batched","contract","output"]:
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    import mujoco
    import supermimic.tasks
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.scene import Scene
    from sonic_g1_runtime import SonicG1Controller,configure_control,verify_compiled_control,rotations
    contract=json.loads(args.contract.read_bytes())
    cfg=load_env_cfg("SuperMimic-GenCoord-G1",play=True);cfg.scene.num_envs=1
    preserved=configure_control(cfg,contract)
    scene=Scene(cfg.scene,"cpu");model=scene.compile();cfg.sim.mujoco.apply(model)
    compiled=verify_compiled_control(model,contract,preserved)
    names=json.loads((args.motions/"joint_names.json").read_bytes())
    policy_names=[r["name"] for r in contract["joints_in_policy_order"]]
    joint_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"robot/"+n) for n in names]
    qidx=[model.jnt_qposadr[j] for j in joint_ids]
    vidx=[model.jnt_dofadr[j] for j in joint_ids]
    policy_qidx=[qidx[names.index(n)] for n in policy_names]
    rows=[]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout="constrained")
    for folder in ["walk1_subject1","full-walk1_subject1","full-run1_subject2"]:
        source=args.pilots/folder
        result=json.loads((source/"result.json").read_bytes())
        assert digest(source/"protocol.json")==result["protocol_sha256"]
        assert digest(source/"trace.npz")==result["trace"]["sha256"]
        with np.load(source/"trace.npz",allow_pickle=False) as trace:
            a={name:trace[name] for name in trace.files}
        n=result["control_steps"]
        assert a["qpos"].shape==(n+1,36) and a["qvel"].shape==(n+1,35)
        assert a["encoder_inputs"].shape==(n,1762) and a["decoder_inputs"].shape==(n,994)
        assert a["actions"].shape==(n,29) and all(np.isfinite(v).all() for v in a.values())
        assert np.array_equal(a["reference_frame"],np.arange(1,n+1))
        assert np.allclose(np.linalg.norm(a["qpos"][:,3:7],axis=-1),1.,atol=1e-10)
        root=np.linalg.norm(a["qpos"][1:,:3]-a["reference_root_pos"][1:],axis=-1)
        joints=np.sqrt(np.mean((a["qpos"][1:,policy_qidx]-a["reference_joint_pos"][1:])**2,axis=-1))
        tilt=2*np.sum(a["qpos"][1:,4:6]**2,axis=-1)
        assert np.allclose(root,a["root_distance"],atol=1e-12,rtol=1e-12)
        assert np.allclose(joints,a["joint_rms"],atol=1e-12,rtol=1e-12)
        assert np.allclose(tilt,a["tilt"],atol=1e-12,rtol=1e-12)
        assert np.isclose(root.mean(),result["mean_root_distance"],atol=1e-12)
        assert np.isclose(joints.mean(),result["mean_joint_rms"],atol=1e-12)
        rows.append(dict(folder=folder,result_sha256=digest(source/"result.json"),
            protocol_sha256=result["protocol_sha256"],trace_sha256=result["trace"]["sha256"],
            steps=n,physical_seconds=n*.02,all_arrays_read_and_finite=True,
            mean_root_distance=float(root.mean()),final_root_distance=float(root[-1]),
            mean_joint_rms_radians=float(joints.mean()),mean_joint_rms_degrees=float(np.rad2deg(joints.mean())),
            maximum_tilt_degrees=float(np.rad2deg(np.arccos(np.clip(1-tilt,-1,1))).max()),
            pilot_stopping_condition_triggered=result["error"] is not None,
            complete_native_reward_termination_evaluator_used=False))
        if folder=="walk1_subject1":
            controller=SonicG1Controller(args.models,args.batched,args.contract,args.motions,names)
            motion_names=json.loads((args.motions/"motion_names.json").read_bytes())
            ends=np.load(args.motions/"motion_ends.npy")
            index=motion_names.index("walk1_subject1");start=int(ends[index-1])
            maximum=np.zeros(3)
            for step in range(n):
                qpos,qvel=a["qpos"][step],a["qvel"][step]
                R=rotations(qpos[3:7])
                state=np.r_[qpos[:7],qpos[qidx],qvel[:3],R@qvel[3:6],qvel[vidx]][None]
                action=controller(state,np.asarray([start+step]))
                differences=[np.max(np.abs(controller.last_encoder_input[0]-a["encoder_inputs"][step])),
                    np.max(np.abs(controller.last_decoder_input[0]-a["decoder_inputs"][step])),
                    np.max(np.abs(action[0,controller.policy_to_native]-a["actions"][step]))]
                maximum=np.maximum(maximum,differences)
                assert np.all(maximum<np.asarray([1e-5,1e-5,1e-4])),(step,maximum)
            parity=dict(states=n,maximum_encoder_input_difference=float(maximum[0]),
                maximum_decoder_input_difference=float(maximum[1]),maximum_native_action_difference=float(maximum[2]))
        else:
            selection=np.unique(np.linspace(0,n-1,800).astype(int));t=(selection+1)*.02
            label=result["motion_name"]
            axes[0].plot(t,root[selection],label=label)
            axes[1].plot(t,np.rad2deg(joints[selection]),label=label)
            axes[2].plot(t,np.rad2deg(np.arccos(np.clip(1-tilt[selection],-1,1))),label=label)
    axes[0].set_ylabel("Root position error (m)")
    axes[1].set_ylabel("Joint RMS error (degrees)")
    axes[2].set_ylabel("Root tilt (degrees)");axes[2].set_xlabel("Simulated time (s)")
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.suptitle("Released SONIC · scalar MuJoCo dynamic pilots\nFull reference sequences; common native reward/termination evaluation still pending",fontsize=12)
    args.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output/"dynamic-pilots.png",dpi=150)
    fig.savefig(args.output/"dynamic-pilots.svg")
    report=dict(status="all_saved_dynamic_pilot_arrays_recomputed_and_common_runtime500state_parity_passed",
        pilots=rows,runtime_parity=parity,compiled29joint_control=compiled,
        GPU_execution_verified=False,paper_results_reproduced=False)
    (args.output/"independent-audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in report.items() if k!="compiled29joint_control"},indent=2))


if __name__=="__main__":main()
