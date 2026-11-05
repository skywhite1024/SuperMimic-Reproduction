"""Read-only model inspection for the user's rollback review; no evaluation/training."""
import gc
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect(label, path, export=None, expected=None):
    p = ROOT / path
    row = {'label': label, 'path': path, 'exists': p.is_file()}
    if not p.is_file():
        return row
    row.update(bytes=p.stat().st_size, sha256=sha(p))
    if expected:
        row['expected_sha256_matches'] = row['sha256'] == expected
    saved = torch.load(p, map_location='cpu', weights_only=True)
    row.update(load_verified=True, iteration=saved.get('iteration'),
               observation_dim=saved['pi']['obs_norm.mean'].numel(),
               state_keys=list(saved),
               wm_present='wm' in saved, critic_present='critic' in saved)
    opts = ['opt_wm', 'opt_pi', 'opt_v']
    row['optimizer_state_counts'] = {
        key: len(saved.get(key, {}).get('state', {})) for key in opts}
    row['full_training_state_present'] = (
        'wm' in saved and 'critic' in saved and all(key in saved for key in opts))
    row['exact_simulator_rng_replay_resume_claimed'] = False
    if export:
        ep = ROOT / export
        exported = torch.load(ep, map_location='cpu', weights_only=True)
        row['export'] = {'path': export, 'bytes': ep.stat().st_size,
                         'sha256': sha(ep), 'load_verified': True,
                         'iteration': exported.get('iteration'),
                         'policy_tensors_equal_full': (
                             saved['pi'].keys() == exported['pi'].keys() and
                             all(torch.equal(v, exported['pi'][k])
                                 for k, v in saved['pi'].items()))}
        del exported
    del saved
    gc.collect()
    return row


def main():
    cases = [
        ('baseline12K', 'implementation/downloaded/runs/baseline-seed0/last.pt', None, None),
        ('baseline16K', 'implementation/continuation-20261025-a/downloaded/policy/full_016000.pt',
         'implementation/continuation-20261025-a/downloaded/policy/ckpt_016000.pt',
         '0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b'),
        ('specialist8808_18K_export', 'interactive-g1-20261027-a/policy/acro-h100-2000.pt', None,
         '0f4a7bc67ae9e3440711e401e37f8276102e59946177117d082a6d7aabe108b4'),
        ('unified22K', 'acro-library-20261027-b/runs/unified-h100-4000/final-full.pt',
         'acro-library-20261027-b/policy/unified-h100-4000.pt',
         '56a6714aa7e79f063f4089c7860a1d00195601f178223bf42b65acd0e587910e'),
        ('unified26K', 'acro-library-20261027-b/runs/unified-h100-stage2-4000/final-full.pt',
         'acro-library-20261027-b/policy/unified-h100-stage2-4000.pt',
         '807d5fac7ca128f892c950fa379133231cd5f741e83075ed017d105b2eec6eec'),
        ('unified30K', 'acro-library-20261027-b/runs/unified-h100-stage3-4000/final-full.pt',
         'acro-library-20261027-b/policy/unified-h100-stage3-4000.pt',
         '93bb8b830145c54092518eba8db1fb020d020eb72cdad1b619b4fc5470cc213d'),
        ('retention31K_rejected', 'acro-library-20261027-b/runs/unified-retention-v1-h100-1000/final-full.pt',
         'acro-library-20261027-b/policy/unified-retention-v1-h100-1000.pt', None),
        ('qualifiedExpertV2_31K_rejected', 'acro-library-20261027-b/runs/unified-qualified-expert-v2-h100-1000/final-full.pt',
         'acro-library-20261027-b/policy/qualified-expert-v2-h100-1000.pt', None),
        ('specialist8807_17K_rejected', 'acro-library-20261027-b/runs/specialist8807-h100-1000-1031d/final-full.pt',
         'acro-library-20261027-b/policy/specialist8807-h100-1000-1031d.pt',
         'f787706f292eb1385e8e78882faf0b3ea4b89e6e992835141eba92c8c9a09758'),
    ]
    models = []
    for case in cases:
        row = inspect(*case)
        models.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    visual = Path(r'[local-user-path]')
    history = json.loads(re.search(
        r'<script type="application/json" id="sm-history-data">(.*?)</script>',
        visual.read_text(encoding='utf-8'), re.S).group(1))
    metrics = {}
    for label, item in history['libraries'].items():
        p = ROOT / item['source']
        actual = json.loads(p.read_text(encoding='utf-8-sig'))
        metrics[label] = {
            'source': item['source'], 'source_sha256': sha(p),
            'backend': item['device'], 'reference': item['reference'],
            'controller': item['controller'],
            'new_hard_gate': actual['groups']['new']['passed'],
            'old_hard_gate': actual['groups'].get('lafan', {}).get('passed'),
            'new_strict': None,
        }
        if 'quality_source' in item:
            qp = ROOT / item['quality_source']
            quality = json.loads(qp.read_text(encoding='utf-8-sig'))
            metrics[label]['quality_source'] = item['quality_source']
            metrics[label]['quality_source_sha256'] = sha(qp)
            # Preserve the exact source schema for audit; do not mix hard/strict.
            metrics[label]['quality_source_summary'] = {
                key: value for key, value in quality.items()
                if key not in ['clips', 'per_clip', 'results', 'failures']
                and not isinstance(value, list)}
            metrics[label]['new_strict'] = item.get('quality', {}).get('combined_engineering_passed')
    continuous = {}
    for label, item in history['continuous'].items():
        cp = ROOT / item['source']
        continuous[label] = dict(item, source_sha256_now=sha(cp))

    result = {
        'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope': 'Local file load/hash/content audit and historical metrics review only; no cloud inspection, new evaluation, rollback or training.',
        'models': models,
        'historical_metrics': metrics,
        'historical_continuous': continuous,
        'limitations': [
            '18K specialist full checkpoint only has historical AFS path/hash; local export freshly verified, remote current existence not rechecked.',
            'Historical 500-update intermediate checkpoints have AFS receipts, not a fresh remote existence guarantee.',
            'Full checkpoint includes model/critic/optimizer state but does not prove exact environment/RNG/replay continuation.',
            '99D baseline/specialists and 101D unified policies need explicit observation compatibility; no direct state-dict interchange.',
            'Original-output historical metrics and bound20 results are separate comparisons; CPU and CUDA are not pooled.',
        ],
        'recommendation': {
            'baseline_and_old_skill_teacher': 'baseline16K',
            'provisional_unified_warm_start_for_raw_new_and_retention': 'unified22K',
            'continuous_and_calibrated_comparison_anchors': ['unified26K', 'unified30K'],
            'do_not_promote': ['retention31K_rejected', 'qualifiedExpertV2_31K_rejected', 'specialist8807_17K_rejected'],
            'not_a_completed_adoption_decision': True,
            'next_minimum_evidence_before_training': 'Compare 22K and 26K under the same present controller/backend and target failure metrics; reuse existing unchanged assessments.',
            'single_next_research_hypothesis': 'Observation-only global XY error feedback for an isolated skill branch; preserve native reward, physics and reference. This is a hypothesis, not a proven fix.',
            'specialist8807_not_qualified_for_distillation': True,
        },
    }
    out = ROOT / 'monitoring/rollback-checkpoint-review-20261101.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('AUDIT_WRITTEN', out, flush=True)


if __name__ == '__main__':
    main()
