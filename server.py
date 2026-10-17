"""Local-only static server and verified-HTTPS market data adapter. Python 3.10+."""
import concurrent.futures
import datetime
import json
import time
import urllib.parse
import urllib.request
import webbrowser
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SECTORS = [('沪深300', '000300'), ('能源', '000928'), ('医药', '000933'),
           ('消费', '000932'), ('制造业', '000930'), ('金融', '000934'), ('科技', '000935')]
ROOT = Path(__file__).resolve().parent


def parse_history(payload, name, code):
    data = payload.get('data')
    if not data or data.get('code') != code:
        raise ValueError(f'{name}：接口未返回指定指数')
    rows = []
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    for line in data.get('klines', []):
        parts = line.split(',')
        date = datetime.date.fromisoformat(parts[0]).isoformat()
        if date > now.date().isoformat():
            raise ValueError(f'{name}：行情日期在未来')
        # Daily candles can be partial during the trading session. Allow a short
        # settlement delay and expose only completed daily observations.
        if date == now.date().isoformat() and now.time() < datetime.time(15, 10):
            continue
        close = float(parts[2])
        if not 0 < close < float('inf'):
            raise ValueError(f'{name}：价格无效')
        rows.append({'date': date, 'name': name, 'close': close})
    if len(rows) < 127 or len({r['date'] for r in rows}) != len(rows):
        raise ValueError(f'{name}：历史行情不足或日期重复')
    return rows


def fetch_sector(sector):
    name, code = sector
    query = urllib.parse.urlencode(dict(
        secid='1.' + code, klt=101, fqt=0, lmt=300, beg='19900101', end='20500101',
        fields1='f1,f2,f3,f4,f5,f6', fields2='f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'))
    req = urllib.request.Request('https://push2his.eastmoney.com/api/qt/stock/kline/get?' + query,
                                 headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'})
    with urllib.request.urlopen(req, timeout=18) as response:
        payload = json.load(response)
    return parse_history(payload, name, code)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == '/api/market':
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
                    rows = [row for group in pool.map(fetch_sector, SECTORS) for row in group]
                self.send_json(200, {'rows': rows, 'source': '东方财富', 'fetched_at': time.time()})
            except Exception as exc:
                print(f'行情获取失败：{type(exc).__name__}: {exc}', flush=True)
                self.send_json(502, {'error': '行情源访问失败或返回数据不完整，请检查网络后重试，也可导入 CSV'})
        elif path in ('/', '/index.html', '/style.css', '/model.js', '/app.js'):
            super().do_GET()
        else:
            self.send_error(404)

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    port = 8765
    server = ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=str(ROOT)))
    print('A股行业观察已启动。关闭此窗口即可停止服务。', flush=True)
    if '--no-browser' not in sys.argv:
        webbrowser.open(f'http://127.0.0.1:{port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
