"""Cache the original public motion descriptions for the five frozen CMU subjects."""
import concurrent.futures,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'library-manifest.json').read_text())
    subjects=sorted({int(Path(row['path']).name.split('_')[0]) for row in manifest['files']})
    def one(subject):
        url=f'https://mocap.cs.cmu.edu/search.php?subjectnumber={subject}';target=ROOT/'sources'/f'cmu-subject-{subject}.html'
        try:
            if not target.exists():
                request=urllib.request.Request(url,headers={'User-Agent':'G1-acro-reproduction-audit'})
                with urllib.request.urlopen(request,timeout=25) as response:target.write_bytes(response.read())
            return {'subject':subject,'source':url,'status':'cached','bytes':target.stat().st_size}
        except Exception as error:return {'subject':subject,'source':url,'status':'unavailable','error':str(error)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(one,subjects))
    (ROOT/'sources/cmu-fetch.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
