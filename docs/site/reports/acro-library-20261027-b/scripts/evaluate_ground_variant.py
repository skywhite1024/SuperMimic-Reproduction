"""Separate physical ablation over explicitly versioned constant-Z references."""
import hashlib,json,sys
from pathlib import Path
from runtime import ROOT,references as raw_references
import evaluate_library


def main():
    manifest=ROOT/'ground-calibration-v1/manifest.json';variant=json.loads(manifest.read_text());assert variant['denominator']==15
    def references(group='all'):
        new=[ROOT/row['path'] for row in variant['files']]
        for path,row in zip(new,variant['files']):assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        return new if group=='new' else raw_references('lafan') if group=='lafan' else new+raw_references('lafan')
    evaluate_library.references=references;evaluate_library.main()
    mode=sys.argv[sys.argv.index('--mode')+1];path=ROOT/'runs'/mode/'summary.json';result=json.loads(path.read_text())
    result.update(reference_variant='ground-calibration-v1',variant_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  raw_library_result=False,controlled_ablation_only=True)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
