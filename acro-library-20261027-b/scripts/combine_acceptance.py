"""Keep tracking, contact and combined engineering acceptance as distinct counts."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();folder=ROOT/'runs'/a.run
    tracking=json.loads((folder/'tracking-assessment.json').read_text());contacts=json.loads((folder/'contact-review.json').read_text())
    samples=json.loads((folder/'contact-samples.json').read_text());contract_path=ROOT/'contact-acceptance-contract.json';contract=json.loads(contract_path.read_text())
    lookup={row['clip']:row for row in contacts['clips']};detail={row['clip']:row['samples'] for row in samples};rows=[]
    for row in tracking['clips']:
        if row['group']!='new':continue
        name=row['clip'];records=detail[name];unexpected=[];window=round(contract['reference_expected_contact_time_tolerance_seconds']/.02)
        for i,record in enumerate(records):
            expected={body for near in records[max(0,i-window):i+window+1] for body in near['reference_ground_bodies']}
            actual={body for body in record['actual_ground_bodies'] if any(word in body.lower() for word in ['head','torso','pelvis'])}
            if actual-expected:unexpected.append({'time_s':record['time_s'],'bodies':sorted(actual-expected)})
        checks={'no_unexpected_trunk_contact':len(unexpected)<=contract['unexpected_head_torso_or_pelvis_ground_contact_samples_max'],
                'penetration_within_engineering_limit':lookup[name]['actual_min_after_initial_100ms_m']>=-contract['actual_ground_penetration_after_initial_100ms_max_m']}
        contact_pass=all(checks.values());rows.append({'clip':name,'original_hard_gate_passed':row['passed_original_hard_gate'],
                'tracking_quality_passed':row['tracking_quality_passed'],'contact_review_passed':contact_pass,'combined_engineering_passed':row['tracking_quality_passed'] and contact_pass,
                'contact_checks':checks,'unexpected_trunk_contact_samples':len(unexpected),'unexpected_trunk_contact_events':unexpected,
                'actual_min_after_initial_100ms_m':lookup[name]['actual_min_after_initial_100ms_m'],'tracking_failed_checks':row['failed_checks'],'contact_force_measured':False})
    assert len(rows)==15;target=folder/'combined-assessment.json';assert not target.exists()
    result={'run':a.run,'new_denominator':15,'tracking_contract_sha256':tracking['contract_sha256'],'contact_contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            'original_hard_gate_passed':sum(x['original_hard_gate_passed'] for x in rows),'tracking_quality_passed':sum(x['tracking_quality_passed'] for x in rows),
            'contact_review_passed':sum(x['contact_review_passed'] for x in rows),'combined_engineering_passed':sum(x['combined_engineering_passed'] for x in rows),'clips':rows}
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({key:value for key,value in result.items() if key!='clips'}))


if __name__=='__main__':main()
