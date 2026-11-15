"""Plot recovered finite PPO-WM curves; excludes unfinished cases and mixed corpora."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
    cases=[]
    for validation in sorted(args.root.glob("ppo-wm-*/validation.json")):
        result=json.loads(validation.read_bytes())
        assert result["status"]=="finite_PPO_WM_batch_training_and_all18_evaluation_originals_verified"
        plan=json.loads((args.root/(validation.parent.name+"-plan.json")).read_bytes())
        label=f"{plan['imagined_batch_size']//1024}k × {plan['steps_per_rollout']}"
        cases.append((label,result))
    assert cases
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained",sharey=True)
    for label,result in cases:
        points=result["points"];returns=[r["mean_normalized_full_motion_return"] for r in points]
        for ax,key in zip(axes,["cumulative_imagined_steps","real_simulator_steps"]):
            ax.plot([r[key] for r in points],returns,"o-",label=label)
            ax.set_xscale("log");ax.grid(alpha=.25);ax.set_ylim(-.01,1.02)
            ax.legend(title="WM batch × horizon")
    axes[0].set_xlabel("Unique imagined transitions (log scale)")
    axes[1].set_xlabel("Real simulator transitions (log scale)")
    axes[1].xaxis.set_major_locator(FixedLocator([2.5e6,1e7,4e7,1e8]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value,position:f"{value/1e6:g}M"))
    for ax in axes:ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Mean normalized full-motion return")
    fig.suptitle(f"Finite PPO-WM · {len(cases)} completed cases · fixed18 clips · seed42\nNative corrected-v2 evaluation; author Figure5 membership/protocol unconfirmed",fontsize=11)
    fig.savefig(args.root/"finite-curves.png",dpi=160)
    fig.savefig(args.root/"finite-curves.svg")
    data=dict(status="only_recovered_verified_cases_plotted",cases=[
        dict(label=label,points=result["points"],training_entry_seconds_to2048=
             result["logged_checkpoints_after_cloud_tensor_and_byte_readback_checks"][-1]["seconds_since_training_entry"])
        for label,result in cases],paper_results_reproduced=False,statistical_multiseed_comparison=False)
    (args.root/"finite-curves-data.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(dict(cases=len(cases),final_points=[(label,result["points"][-1]) for label,result in cases]),indent=2))


if __name__=="__main__":main()
