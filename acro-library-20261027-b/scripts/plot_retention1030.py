"""Plot the complete fixed new-motion denominator and the recovered training log."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    proof = read('runs/unified-retention-v1-h100-1000/proof.json')
    assert proof['new_updates'] == 1000 and proof['world_model_warmup'] == 200
    source = ROOT / 'runs/unified-retention-v1-h100-1000/train_log_with_header.csv'
    with source.open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    iterations = np.array([int(row['iteration']) for row in rows])
    updates = iterations - (iterations[-1] - proof['new_updates'])
    assert np.array_equal(updates, np.arange(50, 1001, 50)), 'The full 1000-update log must be recovered'
    output = ROOT / 'figures'
    output.mkdir(exist_ok=True)
    artifacts = []

    def save(figure, stem):
        for suffix in ['png', 'svg', 'pdf']:
            target = output / f'{stem}.{suffix}'
            figure.savefig(target, dpi=170)
            artifacts.append({'path': target.relative_to(ROOT).as_posix(), 'bytes': target.stat().st_size,
                              'sha256': sha(target)})
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, layout='constrained')
    for axis, field, label, color in zip(axes.flat, ['wm_loss', 'pi_loss', 'sim_rew', 'ep_ret_norm'],
                                       ['World-model loss', 'Native policy loss', 'Training step reward', 'Training episode return'],
                                       ['#325a86', '#a6532e', '#2f7b62', '#785591']):
        values = np.array([float(row[field]) for row in rows])
        if field == 'pi_loss':
            values[updates <= 200] = np.nan
        axis.axvspan(0, 200, color='#e9eef4', label='WM-only warmup')
        axis.plot(updates, values, color=color, lw=1.7)
        axis.set_ylabel(label)
        axis.set_xlim(0, 1000)
        axis.grid(alpha=.18)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel('New updates after the frozen parent checkpoint')
    figure.suptitle('Retention v1: one finite 1000-update candidate', fontsize=15)
    figure.supxlabel('Training metrics do not establish physical acceptance. Teacher penalties are recorded separately in proof.json.', fontsize=8)
    save(figure, 'retention-1030-training')

    names = [Path(row['path']).stem for row in read('library-manifest.json')['files']]
    assert len(names) == 15
    pairs = [
        ('CUDA / raw', 'retention-v1-parent-bound20-raw-cuda-new15', 'retention-v1-raw-cuda-new15'),
        ('CUDA / calibrated', 'retention-v1-parent-bound20-corrected-cuda-all55', 'retention-v1-corrected-cuda-all55'),
        ('CPU / raw', 'stage3-actionbound20-raw-cpu-new15', 'retention-v1-raw-cpu-new15'),
        ('CPU / calibrated', 'stage3-actionbound20-corrected-cpu-all55', 'retention-v1-corrected-cpu-all55'),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True, layout='constrained')
    sources = [{'path': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)}]
    for axis, (label, parent, candidate) in zip(axes.flat, pairs):
        for offset, mode, color, description in [(-.18, parent, '#9aaec1', 'Frozen parent, bound20'),
                                                  (.18, candidate, '#2d7a68', '1000-update candidate')]:
            summary_path = ROOT / f'runs/{mode}/summary.json'
            quality_path = ROOT / f'runs/{mode}/combined-assessment.json'
            physical = {row['clip']: row for row in read(summary_path.relative_to(ROOT))['clips']}
            quality = {row['clip']: row for row in read(quality_path.relative_to(ROOT))['clips']}
            scores = np.array([physical[name]['norm_return_original'] for name in names])
            failed = np.array([not quality[name]['combined_engineering_passed'] for name in names])
            y = np.arange(15) + offset
            axis.barh(y, scores, height=.31, color=color, label=description)
            axis.scatter(scores[failed], y[failed], color='#b52424', marker='x', s=26, zorder=4)
            sources += [{'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)} for path in [summary_path, quality_path]]
        axis.set_title(label)
        axis.axvline(.65, lw=.9, ls='--', color='#55616c')
        axis.set_xlim(-.06, 1.04)
        axis.grid(axis='x', alpha=.15)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_yticks(np.arange(15), [name.replace('_stageii_unitree_g1', '') for name in names])
        axis.set_ylim(14.7, -.7)
    for axis in axes[-1]:
        axis.set_xlabel('Original reward / complete clip denominator')
    axes[0, 0].legend(loc='lower right', framealpha=.95, fontsize=8)
    figure.suptitle('All 15 new motions: matched parent and candidate', fontsize=15)
    figure.supxlabel('Red x: frozen combined tracking/contact criterion failed. One fixed seed per assessment; no uncertainty estimate.', fontsize=9)
    save(figure, 'retention-1030-new15-comparison')
    result = {'status': 'all_fixed_denominators_and_training_rows_plotted', 'new_denominator': 15,
              'comparison_pairs': 4, 'training_rows': len(rows), 'new_updates': 1000, 'sources': sources,
              'artifacts': artifacts, 'acceptance_inferred_from_training_metrics': False}
    (ROOT / 'evidence/retention-1030-plot-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
