"""Inspect explicitly named paused runs without resuming or changing training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metadata(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cases = {
        'raw320k': root / 'runs/bones-raw-320k-1113m',
        'lafan64k': root / 'runs/lafan-curve-64k-1113ag/main/training',
        'ppo_sim': root / 'runs/ppo-sim-2048-1113ak/training',
    }
    report = dict(status='explicit_paused_run_inventory',
        observed_at=datetime.now(timezone.utc).isoformat(), cases={},
        training_resumed=False, model_files_modified=False)
    for label, run in cases.items():
        row = dict(run=str(run), exists=run.is_dir(), originals={})
        parsed = {}
        for name in ('result.json', 'resolved-config.json',
                     'checkpoint-times.json', 'checkpoint-manifest.json'):
            path = run / name
            if path.is_file():
                row['originals'][name] = metadata(path)
                parsed[name] = json.loads(path.read_bytes())
        training = parsed.get('result.json', {})
        row['parent_status'] = training.get('status')
        row['parent_collections'] = training.get('collection_steps')
        row['parent_simulator_steps'] = training.get('simulator_steps')
        row['persisted_PPO_update_rows'] = len(training.get('updates', []))
        if training.get('updates'):
            row['last_PPO_update'] = training['updates'][-1]
        manifests = sorted((run / 'checkpoint-series').glob('step-*/manifest.json'))
        row['retained_native_manifest_count'] = len(manifests)
        if manifests:
            path = manifests[-1]
            row['latest_native_manifest'] = metadata(path)
            manifest = json.loads(path.read_bytes())
            row['latest_retained_collection_steps'] = manifest['collection_steps']
            row['latest_native_models'] = []
            for item in manifest['files']:
                model = path.parent / item['file']
                observed = metadata(model)
                assert observed['bytes'] == item['bytes'] and observed['sha256'] == item['sha256']
                row['latest_native_models'].append(observed)
        ppo_sets = parsed.get('checkpoint-manifest.json', [])
        row['retained_PPO_updates'] = [item['ppo_update'] for item in ppo_sets]
        if ppo_sets:
            row['latest_PPO_models'] = []
            for item in ppo_sets[-1]['files']:
                model = Path(item['path']).resolve()
                assert model.is_relative_to(run / 'checkpoints')
                observed = metadata(model)
                assert observed['bytes'] == item['bytes'] and observed['sha256'] == item['sha256']
                row['latest_PPO_models'].append(observed)
        if label == 'lafan64k':
            row['evaluations'] = []
            for step in (1,8001,16001,24001,32001,40001,48001,56001,64001):
                folder = run.parent / f'eval-{step:09d}'
                point = dict(collection_steps=step, originals={})
                for name in ('result.json','protocol.json','accounting.json',
                             'audit/summary.json','audit/initial-states.json'):
                    path = folder / name
                    if path.is_file():
                        point['originals'][name] = metadata(path)
                point['five_originals_present'] = len(point['originals']) == 5
                if (folder / 'accounting.json').is_file():
                    accounting = json.loads((folder / 'accounting.json').read_bytes())
                    point.update(successes=accounting.get('successes'),
                        mean=accounting.get('mean_normalized_full_motion_return'))
                row['evaluations'].append(point)
        report['cases'][label] = row
        short = dict(label=label, exists=row['exists'], parent_status=row['parent_status'],
            collections=row['parent_collections'], native_manifest_count=len(manifests),
            last_retained_collection=row.get('latest_retained_collection_steps'),
            persisted_PPO_updates=row['persisted_PPO_update_rows'],
            retained_PPO_updates=row['retained_PPO_updates'])
        print('RESUME_INVENTORY_CASE ' + json.dumps(short, separators=(',', ':')), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('RESUME_INVENTORY_COMPLETED ' + json.dumps(metadata(args.output)), flush=True)


if __name__ == '__main__':
    main()
