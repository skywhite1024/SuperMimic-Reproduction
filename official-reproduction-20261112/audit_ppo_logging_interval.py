"""Check pinned native PPO log counters using a synthetic CPU-only workload.

No simulator steps, training or online W&B calls occur. The real PpoRunner.run
and Logger.log execute unchanged; fake collection/learners isolate accounting.
"""
import argparse
import contextlib
import hashlib
import inspect
import io
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import supermimic.runners.ppo_runner as ppo
    import supermimic.utils.logger as logging
    source=Path(inspect.getfile(ppo))
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='04fd4484ca9dd03e86c795d9ffb1e0c584f59754513f39ba824924c5f4e8ab6e'
    updates=2048;real_per_update=10*4096
    runs=[]
    for interval in (1,16):
        calls={'collect':0,'WM':0,'PPO':0,'logs':0}
        logger=logging.Logger('synthetic-log-accounting',1,1,'cpu')
        def collect():
            calls['collect']+=1
            return real_per_update
        def fit(data):
            assert len(list(data))==10
            calls['WM']+=1
            return {}
        def imagine():
            calls['PPO']+=1
            return {}
        def record_log(*arguments,**keywords):calls['logs']+=1
        fake=SimpleNamespace(
            cfg=SimpleNamespace(max_iterations=updates,log_interval=interval,save_interval=updates+1,
                wm=SimpleNamespace(horizon=8,batch_size=2048,num_batches=10)),
            logger=logger,collect_data=collect,imagine=imagine,
            replay_buffer=SimpleNamespace(num_transitions=10,sample=lambda *_:None),
            world_model_learner=SimpleNamespace(update=fit),
            _create_log_dir=lambda *_:Path('.'),_init_wandb=lambda *_,**__:None,
            _seed_sim_history=lambda:None,_cleanup=lambda:None)
        # Each unchanged native iteration observes six clock reads. Its three
        # measured phases are each one synthetic second. This is not runtime.
        ticks=itertools.count()
        with patch.object(ppo.time,'time',side_effect=lambda:float(next(ticks))), \
                patch.object(logging.wandb,'log',side_effect=record_log), \
                contextlib.redirect_stdout(io.StringIO()):
            ppo.PpoRunner.run(fake)
        assert calls['collect']==calls['WM']==calls['PPO']==updates
        expected=updates*real_per_update
        assert calls['logs']==len(range(0,updates,interval))
        assert logger.tot_timesteps==calls['logs']*real_per_update
        assert logger.tot_time==calls['logs']*3
        runs.append(dict(log_interval=interval,actual_native_outer_calls=calls,
            synthetic_collection_steps=expected,native_console_total_steps=logger.tot_timesteps,
            counter_ratio=expected/logger.tot_timesteps,
            synthetic_phase_seconds_all_iterations=updates*3,native_console_time_seconds=logger.tot_time))
    assert runs[0]['counter_ratio']==1 and runs[1]['counter_ratio']==16
    report=dict(status='pinned_PpoRunner_and_Logger_synthetic_counter_mismatch_reproduced',
        workload='CPU-only synthetic collection/learners/clock; unchanged real native outer loop and logger',
        real_simulator_steps_executed=0,training_executed=False,online_WandB_calls=False,
        vendor_files_modified=False,
        source_files={Path(inspect.getfile(mod)).name:hashlib.sha256(Path(inspect.getfile(mod)).read_bytes()).hexdigest() for mod in (ppo,logging)},
        runs=runs,
        implication='At log_interval16, do not use native console Total steps or Time elapsed as cumulative whole-training measurements.',
        finite_job_counters='Use completed native update count times10times4096 for real steps; sum each generated imagined rollout once; use monotonic whole-run elapsed time.',
        current_four_running_payloads_modified=False,paper_results_reproduced=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
