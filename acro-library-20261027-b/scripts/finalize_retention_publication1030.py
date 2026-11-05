"""Record the observed finite publication phase without claiming the research goal."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT.parent / 'publication/SuperMimic-Reproduction'

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))

def main():
    site = read('evidence/retention-1030-pages-deployment-check.json')
    release = read('evidence/retention-1030-release-verification.json')
    download = read('evidence/retention-1030-live-ledger-download-verification.json')
    ui = read('evidence/retention-1030-live-report-playback.json')
    assert site['site']['preview_fix_deployed'] and download['all_nested_records_equal']
    assert len(release['assets']) == 4 and ui['media']['ended'] and ui['continuousMedia']['time'] > 0
    assert ui['continuousMedia']['duration'] == 120.04
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PUBLIC, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main', 'refs/tags/v0.2.0'], cwd=PUBLIC, text=True)
    assert head in remote and release['source_commit'] in remote
    report = PUBLIC / 'docs/site/reports/acro-library-20261027-b/retention-1030-report.html'
    assert hashlib.sha256(report.read_bytes()).hexdigest() == site['site']['sha256']
    result = {'status':'current_finite_candidate_publication_complete', 'checked_at':datetime.now(timezone.utc).isoformat(),
              'source_commit':release['source_commit'], 'site_commit':head, 'tag':'v0.2.0',
              'release_url':release['release_url'], 'report_url':'https://skywhite1024.github.io/SuperMimic-Reproduction/reports/acro-library-20261027-b/retention-1030-report.html',
              'publication_complete':True, 'all4_release_assets_verified':True, 'public_policy_download_and_loader_verified':True,
              'live_filter_search_mode_metrics_playback_and_full_JSON_export_verified':True,
              'public_report_sha256':site['site']['sha256'], 'candidate_adopted':False, 'full_goal_complete':False,
              'historical_v0_1_0_unchanged':True, 'new_GPU_jobs':0,
              'resource_release_evidence':'evidence/retention-resume1030b-final-resource-verification.json',
              'resource_release_observation_local':'2026-10-30 14:48:30 +08:00',
              'later_resource_UI_refresh':'Existing stale tab did not respond; no new cloud action in this publication phase',
              'original_attempt_a_console_export_complete':False,
              'UI_download_event_timed_out_but_actual_file_verified':True,
              'next':'Execute the frozen zeroGPU bounded stage2 teacher qualification; no automatic new GPU block'}
    (ROOT/'evidence/retention-1030-Git-publication.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    execution = read('execution.json')
    execution.update(stage='retention_1030_publication_complete_next_zero_GPU_teacher_qualification', updated_at=datetime.now(timezone.utc).isoformat(),full_goal_complete=False)
    execution['publication'] = result
    (ROOT/'execution.json').write_text(json.dumps(execution,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    handoff = {'updated_at':result['checked_at'],'active_task':result['next'],'current_candidate_adopted':False,
               'full_goal_complete':False,'publication':result,'local_render_and_video_preparation':'all complete',
               'cloud_resources_active_by_this_phase':False,'next_plan':'evidence/retention-1030-next-diagnostic-plan.json',
               'preserve_BONES_pause':True,'no_new_GPU_jobs_automatically_authorized':True}
    (ROOT/'evidence/retention-1030-active-handoff.json').write_text(json.dumps(handoff,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
