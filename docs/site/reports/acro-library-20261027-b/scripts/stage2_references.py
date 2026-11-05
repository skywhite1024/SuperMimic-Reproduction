"""The explicitly labelled constant-Z reference variant, preserving all 55 clips."""
import hashlib,json
from runtime import ROOT,references as raw_references

def references(group='all'):
    manifest=ROOT/'ground-calibration-v1/manifest.json'
    plan=json.loads((ROOT/'stage2-plan.json').read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==plan['files']['ground-calibration-v1/manifest.json']
    variant=json.loads(manifest.read_text());assert variant['denominator']==15 and variant['modified_clips']==2
    new=[]
    for entry in variant['files']:
        path=ROOT/entry['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],entry['clip'];new.append(path)
    old=raw_references('lafan')
    return new if group=='new' else old if group=='lafan' else new+old

