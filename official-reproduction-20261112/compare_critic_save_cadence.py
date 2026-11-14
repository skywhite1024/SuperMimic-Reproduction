"""Describe all40 H32 endpoints at the same 4k model-save cadence."""
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    root=Path(__file__).parent
    critic=root/'lafan-curve-64k-1115b'
    no=root/'critic-free-main-1113ah'
    files=[critic/'eval-000016001/result.json',no/'eval-000016001/result.json',
           critic/'eval-000016001/protocol.json',no/'eval-000016001/protocol.json',
           critic/'training/resolved-config.json',no/'training/resolved-config.json']
    with_result,without_result,with_protocol,without_protocol,with_cfg,without_cfg=[json.loads(path.read_bytes()) for path in files]
    assert with_protocol['motion_files']==without_protocol['motion_files']
    assert with_protocol['manual_reset_version']==without_protocol['manual_reset_version']==2
    for cfg in (with_cfg,without_cfg):
        assert cfg['seed']==42 and cfg['num_envs']==4096
        assert cfg['runner']['save_interval']==4000
        assert cfg['runner']['policy']['horizon']==32 and cfg['runner']['policy']['gamma']==.99
        assert cfg['training_profile']=='paper-a5-single-gpu' and cfg['goal_assembly_backend']=='eager'
    assert 'lam' in with_cfg['runner']['policy'] and 'lam' not in without_cfg['runner']['policy']
    with_by_index={row['index']:row for row in with_result['records']}
    assert set(with_by_index)==set(range(40))
    rows=[]
    for after in without_result['records']:
        before=with_by_index[after['index']]
        assert before['name']==after['name'] and before['reference_frames']==after['reference_frames']
        x=before['reward_sum']/before['reference_frames']/.1
        y=after['reward_sum']/after['reference_frames']/.1
        rows.append(dict(index=before['index'],name=before['name'],reference_frames=before['reference_frames'],
            critic_success=before['success'],no_critic_success=after['success'],
            critic_return=x,no_critic_return=y,no_critic_minus_critic=y-x))
    assert len(rows)==40 and len({row['index'] for row in rows})==40
    means=[sum(row[key] for row in rows)/40 for key in ('critic_return','no_critic_return')]
    assert math.isclose(means[0],.7858569384223237,rel_tol=1e-13)
    assert math.isclose(means[1],.6354776196957028,rel_tol=1e-13)
    groups={f'critic_{a}_no_critic_{b}':sum(row['critic_success']==a and row['no_critic_success']==b for row in rows)
            for a in (True,False) for b in (True,False)}
    result=dict(status='descriptive_matched_save_cadence_H32_full40_comparison',
        training_collection_steps=16001,save_interval=4000,num_envs=4096,seed=42,
        full40_reference_frames=sum(row['reference_frames'] for row in rows),
        critic_successes=sum(row['critic_success'] for row in rows),
        no_critic_successes=sum(row['no_critic_success'] for row in rows),
        critic_mean=means[0],no_critic_mean=means[1],no_critic_minus_critic_mean=means[1]-means[0],
        per_clip_higher_without_critic=sum(row['no_critic_minus_critic']>0 for row in rows),
        per_clip_lower_without_critic=sum(row['no_critic_minus_critic']<0 for row in rows),
        success_transitions=groups,most_negative_deltas=sorted(rows,key=lambda row:row['no_critic_minus_critic'])[:5],
        multiple_seeds=False,training_episode_return_curve=False,
        exact_wall_time_comparison=False,causal_effect_or_paper_results_established=False,
        sources={str(path.relative_to(root)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    output=critic/'matched-critic-h32-16k';output.mkdir(exist_ok=True)
    with (output/'per-clip.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(sorted(rows,key=lambda row:row['index']))
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','most_negative_deltas')},indent=2))


if __name__=='__main__':main()
