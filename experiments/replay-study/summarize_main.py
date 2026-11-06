"""Analyze only a complete matched-backend 12-run main matrix; retain seed points."""
import argparse
import json
from pathlib import Path
import statistics
from study_runtime import ROOT, STUDY, plan, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); spec = plan()
    assert spec['main_budget_frozen'], 'Engineering/pilot outputs are not main results'
    receipt = json.loads(a.receipt.read_text(encoding='utf-8'))
    assert receipt['status'] == 'all_13_full55_evaluations_completed'
    assert receipt['plan_sha256'] == sha(STUDY/'plan.json')
    assert receipt['device'] == 'cuda' and receipt['backend'] == 'mjwarp'
    expected = {f'rho{round(rho*100):03d}-seed{seed}':(rho,seed)
                for rho in spec['old_replay_fractions'] for seed in spec['training_seeds']}
    assert len(receipt['cases']) == 13 and {c['case'] for c in receipt['cases']} == {'baseline',*expected}
    summaries = {}; evaluation_seconds = 0.
    for case in receipt['cases']:
        source = ROOT/case['summary']
        assert source.resolve().is_relative_to(ROOT.resolve()) and sha(source) == case['summary_sha256']
        result = json.loads(source.read_text(encoding='utf-8'))
        assert result['backend'] == 'mjwarp' and result['observation_dim'] == 99
        assert result['post_initialization_resets'] == 0 and len(result['clips']) == 55
        for group, count in [('lafan',40),('new',15)]:
            clips = [c for c in result['clips'] if c['group'] == group]
            assert len(clips) == count == result['groups'][group]['denominator']
            assert abs(statistics.mean(c['norm_return_original'] for c in clips)
                       - result['groups'][group]['macro_return_original']) < 1e-6
        summaries[case['case']] = result
        evaluation_seconds += result['wall_seconds']
    baseline = summaries['baseline']
    assert baseline['checkpoint_sha256'] == spec['parent_sha256']
    points = []; training_seconds = 0.
    for tag,(rho,seed) in expected.items():
        result = summaries[tag]
        proof = json.loads((STUDY/'runs'/f'{a.prefix}-{tag}'/'proof.json').read_text(encoding='utf-8'))
        assert proof['scientific_main_result'] and proof['device'] == 'cuda'
        assert proof['new_updates'] == spec['provisional_main']['steps']
        assert proof['variant']['plan_sha256'] == receipt['plan_sha256']
        assert proof['variant']['seed'] == seed and proof['variant']['old_replay_fraction'] == rho
        assert result['checkpoint_sha256'] == proof['checkpoint_sha256']
        old = result['groups']['lafan']; new = result['groups']['new']
        points.append({'rho':rho,'seed':seed,'old_return':old['macro_return_original'],
            'additional_return':new['macro_return_original'],
            'forgetting':baseline['groups']['lafan']['macro_return_original']-old['macro_return_original'],
            'adaptation_gain':new['macro_return_original']-baseline['groups']['new']['macro_return_original'],
            'old_original_gate_failures':40-old['passed'],'additional_original_gate_failures':15-new['passed'],
            'old_body_local_mean_m':statistics.mean(c['body_local_mean_m'] for c in result['clips'] if c['group']=='lafan'),
            'additional_body_local_mean_m':statistics.mean(c['body_local_mean_m'] for c in result['clips'] if c['group']=='new')})
        training_seconds += proof['wall_seconds']
    aggregates = []
    metrics = [key for key in points[0] if key not in ['rho','seed']]
    for rho in spec['old_replay_fractions']:
        rows = [r for r in points if r['rho']==rho]; assert len(rows)==3
        aggregates.append({'rho':rho,'n_training_seeds':3,
            'metrics':{key:{'mean':statistics.mean(r[key] for r in rows),
                            'sample_std':statistics.stdev(r[key] for r in rows)} for key in metrics}})
    output = a.output.resolve(); assert not output.exists(); output.mkdir(parents=True)
    analysis = {'baseline':baseline['groups'],'seed_points':points,'conditions':aggregates,
        'program_gpu_hours_training_and_evaluation':(training_seconds+evaluation_seconds)/3600,
        'timing_excludes':['queue time','source preparation','probe diagnostics','failed preparation jobs'],
        'limitations':['Three continuation seeds share one pretrained parent.',
            'Returns are not success probabilities; original gates do not certify strict physical quality.',
            'Within-library adaptation and retention do not establish unseen-motion generalization.']}
    (output/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(6.5,4.5))
    for group in aggregates:
        rho = group['rho']; m = group['metrics']; rows=[r for r in points if r['rho']==rho]
        plot = ax.errorbar(m['forgetting']['mean'],m['adaptation_gain']['mean'],
            xerr=m['forgetting']['sample_std'],yerr=m['adaptation_gain']['sample_std'],
            fmt='o',capsize=3,label=f'old replay {rho:g}')
        ax.scatter([r['forgetting'] for r in rows],[r['adaptation_gain'] for r in rows],
                   color=plot[0].get_color(),marker='x',alpha=.7)
    ax.axvline(0,color='gray',linewidth=.8); ax.axhline(0,color='gray',linewidth=.8)
    ax.set(xlabel='Old-motion forgetting (baseline minus continued return)',
           ylabel='Additional-motion adaptation gain',title='Replay trade-off: 3 continuation seeds per ratio')
    ax.legend(); fig.tight_layout()
    fig.savefig(output/'replay-tradeoff.png',dpi=180); fig.savefig(output/'replay-tradeoff.pdf'); plt.close(fig)
    print('COMPLETE_MAIN_MATRIX_ANALYZED '+str(output),flush=True)


if __name__ == '__main__': main()
