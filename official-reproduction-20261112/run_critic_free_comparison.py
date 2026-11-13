"""Bounded H32 no-critic CUDA integration, then a fresh16k full40 endpoint."""
import argparse
from pathlib import Path
from run_lafan_learning_curve import run_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    motions=args.motions.resolve()
    run_case(output/'integration',motions,21,10,[11],900,600,
             policy_horizon=32,without_critic=True)
    print('NO_CRITIC_CUDA_INTEGRATION_PASSED',flush=True)
    run_case(output/'main',motions,16001,4000,[16001],5400,900,
             policy_horizon=32,without_critic=True)
    print('NO_CRITIC_16K_PIPELINE_COMPLETED',flush=True)


if __name__=='__main__':
    main()
