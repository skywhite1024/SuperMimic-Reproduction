"""Select an intact, pre-stop logged snapshot without declaring training complete."""
import json
from pathlib import Path

from select_completed_checkpoint import digest


def select_interrupted_snapshot(run, collection_step, evidence_path):
    run = Path(run).resolve()
    evidence_path = Path(evidence_path).resolve()
    evidence = json.loads(evidence_path.read_bytes())
    if evidence['parent_platform_status'] != 'stopped_by_user':
        raise ValueError('Explicit user-stop evidence is required')
    if str(run) != evidence['original_training_directory']:
        raise ValueError('Interrupted parent path differs from logged evidence')
    expected = next(r for r in evidence['logged_snapshots'] if r['collection_steps'] == collection_step)
    folder = run / 'checkpoint-series' / f'step-{collection_step:09d}'
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if manifest != expected or manifest['status'] != 'model_snapshot_preserved':
        raise ValueError('Saved manifest differs from the immutable pre-stop log')
    if manifest['full_training_resume_available'] is not False:
        raise ValueError('Unexpected resume contract')
    required = {'world_model.pt', 'actor.pt', 'critic.pt'}
    records = manifest['files']
    if len(records) != 3 or {r['file'] for r in records} != required:
        raise ValueError('Exactly three logged native weights are required')
    checkpoints = {}
    for row in records:
        path = folder / row['file']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Interrupted checkpoint size or SHA256 differs: ' + path.name)
        checkpoints[path.name] = path
    resolved = json.loads((run / 'resolved-config.json').read_bytes())
    if resolved['corpus']['name'] != 'lafan-locomotion18' or resolved['seed'] != 42:
        raise ValueError('Unexpected interrupted training corpus/seed')
    training = dict(status='interrupted_parent_native_saved_weights_only',
        originally_planned_collection_steps=resolved['collection_steps'],
        last_pre_stop_logged_profile_collection=evidence['last_logged_profile_collection'],
        evaluated_collection_steps=collection_step, parent_training_complete=False,
        full_training_resume_available=False, paper_results_reproduced=False)
    selection = dict(kind='interrupted_parent_pre_stop_logged_snapshot',
        evaluated_collection_steps=collection_step, parent_training_complete=False,
        parent_stop_evidence_sha256=digest(evidence_path),
        parent_training_directory=str(run), manifest_sha256=digest(manifest_path),
        files=[dict(path=str(checkpoints[r['file']]), bytes=r['bytes'], sha256=r['sha256']) for r in records],
        full_training_resume_available=False)
    return training, checkpoints, selection
