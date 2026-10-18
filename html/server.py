"""Local dashboard server: scheduled close-data refresh and saved regression inference."""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path
import threading
import urllib.parse
import webbrowser

from daily.service import DailyService

HTML=Path(__file__).resolve().parent
STATIC={'/','/index.html','/style.css','/model.js','/app.js','/snapshot.js'}


def handler_for(service):
    class Handler(SimpleHTTPRequestHandler):
        def allowed_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

        def do_GET(self):
            if not self.allowed_host():
                self.send_error(403);return
            path=urllib.parse.urlsplit(self.path).path
            if path=='/api/dashboard':
                self.send_json(200,service.dashboard())
            elif path in STATIC:
                super().do_GET()
            else:
                self.send_error(404)

        def do_HEAD(self):
            if not self.allowed_host() or urllib.parse.urlsplit(self.path).path not in STATIC:
                self.send_error(404);return
            super().do_HEAD()

        def do_POST(self):
            origin=self.headers.get('Origin')
            allowed_origins={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if not self.allowed_host() or (origin and origin not in allowed_origins):
                self.send_error(403);return
            if self.path!='/api/refresh':
                self.send_error(404);return
            launch_refresh(service)
            self.send_json(202,{'refreshing':True})

        def send_json(self,status,payload):
            body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)

    return partial(Handler,directory=str(HTML))


def launch_refresh(service):
    def job():
        try:
            if service.refresh():service.export_snapshot()
        except Exception as exc:
            print(f'行情更新未完成，保留原有数据：{exc}',flush=True)
    threading.Thread(target=job,daemon=True).start()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--offline',action='store_true',help='启动时不联网；仅手动刷新')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    service=DailyService()
    service.export_snapshot()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(service))
    stop=threading.Event()
    if not args.offline:
        launch_refresh(service)
        def scheduled():
            while not stop.wait(1800):launch_refresh(service)
        threading.Thread(target=scheduled,daemon=True).start()
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'A股行业观察：{url}；关闭此窗口停止每日更新。',flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:stop.set();server.server_close()


if __name__=='__main__':main()
