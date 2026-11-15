"""Recompute the finite horizon endpoint comparison and plot verified local profiles."""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / 'finite-comparison-1115'
    output.mkdir(exist_ok=True)
    horizon_rows, paired_rows, sources = [], [], []
    for horizon in (8, 16, 32, 64, 128):
        variants = {}
        for critic in (True, False):
            if horizon == 32:
                folder = ROOT / ('lafan-curve-64k-1115b' if critic else 'critic-free-main-1113ah')
            else:
                job = 'critic-grid-h8h16-1115f' if horizon < 32 else 'critic-grid-h64h128-1115g'
                folder = ROOT / job / (f'h{horizon:03d}-' + ('critic' if critic else 'no-critic')) / 'main'
            evaluation = folder / 'eval-000016001'
            result, protocol, accounting = [read(evaluation / name) for name in ('result.json', 'protocol.json', 'accounting.json')]
            training = read(folder / 'training/result.json')
            cfg = read(folder / 'training/resolved-config.json')
            assert cfg['runner']['policy']['horizon'] == horizon and cfg['runner']['save_interval'] == 4000
            assert cfg['seed'] == 42 and cfg['num_envs'] == 4096
            assert protocol['clips'] == result['clips'] == len(result['records']) == 40
            assert accounting['reference_frames'] == 441140
            assert protocol['checkpoint_selection']['evaluated_collection_steps'] == 16001
            assert accounting['result_sha256'] == digest(evaluation / 'result.json')
            assert protocol['checkpoint_selection']['parent_result_sha256'] == digest(folder / 'training/result.json')
            rows = {row['index']: row for row in result['records']}
            assert set(rows) == set(range(40))
            mean = sum(row['reward_sum'] / row['reference_frames'] / .1 for row in rows.values()) / 40
            assert math.isclose(mean, accounting['mean_normalized_full_motion_return'], rel_tol=1e-13)
            assert sum(row['success'] for row in rows.values()) == accounting['successes']
            source = dict(horizon=horizon, critic=critic, folder=folder.relative_to(ROOT).as_posix(),
                          result_sha256=digest(evaluation / 'result.json'),
                          protocol_sha256=digest(evaluation / 'protocol.json'),
                          training_parent_collections=training['collection_steps'])
            sources.append(source)
            horizon_rows.append(dict(horizon=horizon, critic_enabled=critic, evaluated_collections=16001,
                                     real_simulator_steps=16001 * 4096, successes=accounting['successes'],
                                     failures=accounting['failures'], mean_normalized_full_motion_return=mean))
            variants[critic] = rows
        for index in range(40):
            a, b = variants[True][index], variants[False][index]
            assert (a['name'], a['reference_frames']) == (b['name'], b['reference_frames'])
            score = lambda row: row['reward_sum'] / row['reference_frames'] / .1
            paired_rows.append(dict(horizon=horizon, index=index, motion=a['name'],
                                    reference_frames=a['reference_frames'], critic_success=a['success'],
                                    no_critic_success=b['success'], critic_return=score(a),
                                    no_critic_return=score(b), no_critic_minus_critic=score(b) - score(a)))
    for filename, rows in [('horizon-endpoints.csv', horizon_rows), ('horizon-paired-clips.csv', paired_rows)]:
        with (output / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    ppo = read(ROOT / 'ppo-sim-2048-1115c/validation.json')
    native = read(ROOT / 'lafan-curve-64k-1115b/validation.json')
    assert ppo['scientific_training_and_evaluation_complete'] and ppo['original_JSON'] == 39
    assert native['all_original_byte_counts_and_SHA256_verified']
    native_points = native['points']
    curve_rows = []
    for algorithm, points in [('Native H32 critic', native_points), ('PPO-Sim local profile', ppo['main_points'])]:
        for point in points:
            steps = point['collection_steps'] * 4096 if algorithm.startswith('Native') else point['real_simulator_steps']
            curve_rows.append(dict(algorithm=algorithm, real_simulator_steps=steps,
                                  mean_normalized_full_motion_return=point['mean_normalized_full_motion_return'],
                                  successes=point['successes']))
    with (output / 'finite-learning-points.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]), lineterminator='\n');writer.writeheader();writer.writerows(curve_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    colors = ('#2467a8', '#c66a20')
    for critic, color, marker in [(True, colors[0], 'o'), (False, colors[1], 's')]:
        rows = [row for row in horizon_rows if row['critic_enabled'] == critic]
        axes[0].plot([row['horizon'] for row in rows], [row['mean_normalized_full_motion_return'] for row in rows],
                     color=color, marker=marker, linewidth=2, label='With critic' if critic else 'Without critic')
    axes[0].set_xscale('log', base=2);axes[0].set_xticks([8, 16, 32, 64, 128], labels=['8', '16', '32', '64', '128'])
    axes[0].set_xlabel('Policy horizon');axes[0].set_title('Matched 16,001-collection endpoints')
    for algorithm, color in zip(('Native H32 critic', 'PPO-Sim local profile'), colors):
        rows = [row for row in curve_rows if row['algorithm'] == algorithm]
        axes[1].plot([row['real_simulator_steps'] / 1e6 for row in rows],
                     [row['mean_normalized_full_motion_return'] for row in rows],
                     color=color, marker='o', linewidth=2, label=algorithm)
    axes[1].set_xlabel('Real simulator transitions (millions)');axes[1].set_title('Finite learning curves on the same 40 motions')
    for axis in axes:
        axis.set_ylim(0, 1);axis.set_ylabel('Mean normalized full-motion return')
        axis.grid(alpha=.2);axis.legend(loc='best', frameon=False)
    fig.suptitle('Qualified local SuperMimic results — seed 42', fontsize=14)
    fig.text(.5, .02, 'Fixed-start evaluation on training motions; H32 snapshots reused. PPO hyperparameters are not author-confirmed.\n'
             'Single seed; no paper-equivalent speedup, held-out generalization or hardware claim.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    for suffix in ('png', 'svg'):
        fig.savefig(output / ('finite-comparison.' + suffix), dpi=180)
    plt.close(fig)
    summary = dict(status='ten_horizon_endpoints_and_finite_learning_curves_recomputed', seed=42,
                   horizon_cases=10, paired_motion_comparisons=200, full40_reference_frames=441140,
                   horizon_endpoints=horizon_rows, sources=sources, PPO_original_exit_code=ppo['original_job_exit_code'],
                   PPO_original_export_failure_preserved=True, PPO_scientific_training_and_evaluation_complete=True,
                   first_native_observed_sample_at_or_above_082=native['first_observed_sample_at_or_above_082'],
                   PPO_observed_sample_at_or_above_082=None, PPO_best_observed_mean=max(row['mean_normalized_full_motion_return'] for row in ppo['main_points']),
                   paper_PPO_hyperparameters_confirmed=False, paper_training_episode_curves_completed=False,
                   multiple_seeds=False, causal_effect_established=False, paper_results_reproduced=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(status=summary['status'], horizon_cases=10, paired_clips=200,
                         figure=str(output / 'finite-comparison.png')), indent=2))


if __name__ == '__main__':
    main()
