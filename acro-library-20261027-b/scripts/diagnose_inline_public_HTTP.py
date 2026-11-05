"""One bounded public HEAD probe; records TLS category without credentials."""
import json,ssl,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
receipt=json.loads((root/'evidence/qualified-expert-v2-inline-source-publication.json').read_text())
url=f"https://raw.githubusercontent.com/skywhite1024/SuperMimic-Reproduction/{receipt['commit']}/artifacts/qualified-expert-v2-inline-init/qualified-expert-v2-inline-init-envelope.tar.gz"
proof={'transport':'Python_stdlib_HEAD','URL':url,'TLS_verification_disabled':False}
try:
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=15) as response:proof['HTTP_status']=response.status
except Exception as error:
    reason=getattr(error,'reason',error);proof.update(error_type=type(error).__name__,reason_type=type(reason).__name__)
    if isinstance(reason,ssl.SSLCertVerificationError):proof.update(verify_code=reason.verify_code,verify_message=reason.verify_message)
(root/'evidence/qualified-expert-v2-inline-public-HTTP-Python-diagnostic.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
