"""Fresh native first-order learning on the same18 motions as the PPO-WM sweep."""
import argparse
from pathlib import Path
import subprocess
import sys

from run_lafan_learning_curve import POINTS,execute,run_case

SCRIPTS=Path(__file__).parent
FILES=('result.json','protocol.json','accounting.json','audit/summary.json','audit/initial-states.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    try:
        execute('prepare_lafan_locomotion18.py',['--source',args.source_motions,'--output',output/'motions'],output/'prepare.log',180)
        run_case(output/'main',output/'motions',64001,4000,POINTS,14400,900,
                 emit_receipts=False,corpus='lafan-locomotion18',record_learner_counts=True)
    finally:
        # Preserve failed/partial outputs too; transport never triggers retraining.
        files=[output/'motion-subset.json']
        files += [output/'main/training'/name for name in ('result.json','resolved-config.json','checkpoint-times.json','learner-counts.json')]
        files += [output/'main'/f'eval-{step:09d}'/name for step in POINTS for name in FILES]
        files += [output/'main/result.json']
        files=[p for p in files if p.is_file()]
        print('LOCOMOTION18_CONTROL_ORIGINALS_BEGIN',flush=True)
        subprocess.run([sys.executable,str(SCRIPTS/'emit_json_receipts.py'),'--max-bytes',64000000,*map(str,files)],check=True)
        print('LOCOMOTION18_CONTROL_ORIGINALS_END',flush=True)
    print('LOCOMOTION18_CONTROL_COMPLETED',flush=True)


if __name__=='__main__':main()
