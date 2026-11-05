"""Report the unchanged single-clip strict gate, including every failed tail."""
import argparse,hashlib,json
from runtime import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();folder=ROOT/'runs'/a.run
    summary=json.loads((folder/'summary.json').read_text());tracking=json.loads((folder/'tracking-assessment.json').read_text());contact=json.loads((folder/'contact-review.json').read_text());samples=json.loads((folder/'contact-samples.json').read_text())
    contract=json.loads((ROOT/'contact-acceptance-contract.json').read_text());assert tracking['new_denominator']==contact['new_denominator']==len(samples)==1
    records=samples[0]['samples'];window=round(contract['reference_expected_contact_time_tolerance_seconds']/.02);unexpected=0
    for i,row in enumerate(records):
        expected={body for near in records[max(0,i-window):i+window+1] for body in near['reference_ground_bodies']}
        actual={body for body in row['actual_ground_bodies'] if any(word in body.lower() for word in ['head','torso','pelvis'])}
        unexpected+=bool(actual-expected)
    t=tracking['clips'][0];c=contact['clips'][0]
    result={'run':a.run,'clip':'88_07_stageii_unitree_g1','tracking':t,'unexpected_trunk_contact_samples':unexpected,
            'actual_min_after_initial_100ms_m':c['actual_min_after_initial_100ms_m'],
            'strict_passed':bool(t['tracking_quality_passed'] and unexpected<=contract['unexpected_head_torso_or_pelvis_ground_contact_samples_max'] and c['actual_min_after_initial_100ms_m']>=-contract['actual_ground_penetration_after_initial_100ms_max_m']),
            'checkpoint_sha256':summary['checkpoint_sha256'],'reference_variant':summary['reference_variant'],
            'tracking_contract_sha256':hashlib.sha256((ROOT/'acceptance-contract.json').read_bytes()).hexdigest(),
            'contact_contract_sha256':hashlib.sha256((ROOT/'contact-acceptance-contract.json').read_bytes()).hexdigest(),
            'full_physical_tail_retained':True,'unified55_goal_complete':False}
    (folder/'specialist-assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
