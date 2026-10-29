"""Loopback-only report preview with byte ranges for browser video seeking."""
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(ROOT),**kwargs)

    def end_headers(self):
        self.send_header('Accept-Ranges','bytes')
        super().end_headers()

    def send_head(self):
        self.remaining=None
        file=Path(self.translate_path(self.path))
        if not file.resolve().is_relative_to(ROOT):
            self.send_error(403);return None
        header=self.headers.get('Range')
        if file.is_file() and header:
            match=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
            size=file.stat().st_size
            if not match:
                self.send_error(416);return None
            start=int(match[1]);end=int(match[2]) if match[2] else size-1
            end=min(end,size-1)
            if not 0<=start<=end<size:
                self.send_error(416);return None
            stream=file.open('rb');stream.seek(start);self.remaining=end-start+1
            self.send_response(206);self.send_header('Content-Type',self.guess_type(str(file)))
            self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.send_header('Content-Length',str(self.remaining));self.end_headers();return stream
        return super().send_head()

    def copyfile(self,source,outputfile):
        if self.remaining is None:return super().copyfile(source,outputfile)
        remaining=self.remaining
        while remaining:
            data=source.read(min(1<<20,remaining))
            if not data:break
            outputfile.write(data);remaining-=len(data)


if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',64727),Handler).serve_forever()
