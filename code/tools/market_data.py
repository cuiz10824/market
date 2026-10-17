"""Fetch, organize and audit index history. Python 3.10+, standard library only."""
import argparse
import csv
import concurrent.futures
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import struct
import sys
import tarfile
import urllib.request
import urllib.parse
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
FIELDS = ['date', 'code', 'name', 'close']
RELEASE = '2026-10-17'
SOURCE = 'https://github.com/chenditc/investment_data'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if not path.exists() or path.read_text(encoding='utf-8') != text:
        path.write_text(text, encoding='utf-8')


def registry():
    return json.loads((DATA / 'registry.json').read_text(encoding='utf-8'))


def validate_rows(rows):
    if not rows:
        raise ValueError('empty price series')
    previous = ''
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    for row in rows:
        date = row['date']
        if dt.date.fromisoformat(date).isoformat() != date or date > today:
            raise ValueError('invalid or future observation date')
        if date <= previous:
            raise ValueError('duplicate or unordered dates')
        if dt.date.fromisoformat(date).weekday() >= 5:
            raise ValueError('unexpected weekend observation')
        close = float(row['close'])
        if not math.isfinite(close) or close <= 0:
            raise ValueError('invalid close')
        previous = date
    return rows


def eastmoney_url(item):
    return ('https://push2his.eastmoney.com/api/qt/stock/kline/get?'
            f"secid={item['secid']}&klt=101&fqt=0&lmt=10000&end=20500101"
            '&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61')


def parse_eastmoney(payload, item):
    data = payload.get('data')
    if payload.get('rc') != 0 or not data or data.get('code') != item['code']:
        raise ValueError('missing or mismatched index code')
    if data.get('name') != item['name']:
        raise ValueError(f"unexpected index name: {data.get('name')}")
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    result = []
    for line in data.get('klines', []):
        values = line.split(',')
        if len(values) < 7:
            raise ValueError('incomplete daily candle')
        date = values[0]
        if date == now.date().isoformat() and now.time() < dt.time(15, 10):
            continue
        o, c, h, lo = map(float, values[1:5])
        if not all(math.isfinite(v) and v > 0 for v in (o, c, h, lo)):
            raise ValueError('invalid OHLC')
        if lo > min(o, c) or h < max(o, c) or lo > h:
            raise ValueError('inconsistent OHLC')
        result.append({'date': date, 'code': item['code'], 'name': item['name'], 'close': values[2]})
    return validate_rows(result)


def qlib_rows(archive, symbol, item):
    """Use preserved adjclose, never the rebased close feature as index points."""
    days = archive.read('calendars/day.txt').decode().splitlines()
    raw = archive.read(f'features/{symbol}/adjclose.day.bin')
    if len(raw) % 4 or len(raw) < 8:
        raise ValueError('invalid Qlib feature byte length')
    values = struct.unpack('<' + 'f' * (len(raw) // 4), raw)
    start = values[0]
    if not math.isfinite(start) or start != int(start) or start < 0:
        raise ValueError('invalid Qlib calendar offset')
    start = int(start)
    if start + len(values) - 1 > len(days):
        raise ValueError('feature extends beyond observed calendar')
    rows = []
    for i, value in enumerate(values[1:]):
        if math.isnan(value):
            continue  # Preserve real missing observations; never forward-fill.
        rows.append({'date': days[start + i], 'code': item['code'], 'name': item['name'],
                     'close': f'{value:.2f}'})
    return validate_rows(rows)


def tencent_symbol(item):
    return ('sz' if item['secid'].startswith('0.') else 'sh') + item['code']


def parse_tencent(snapshot, item):
    symbol = tencent_symbol(item)
    if snapshot.get('symbol') != symbol:
        raise ValueError('Tencent snapshot symbol mismatch')
    rows = {}
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    for window in snapshot['windows']:
        raw = window['response_text'].encode('utf-8')
        if hashlib.sha256(raw).hexdigest() != window['sha256']:
            raise ValueError('Tencent raw response digest mismatch')
        payload = json.loads(raw)
        if payload.get('code') != 0 or not isinstance(payload.get('data'), dict):
            raise ValueError('Tencent endpoint returned error or empty symbol map')
        data = payload['data'].get(symbol)
        if not isinstance(data, dict):
            raise ValueError('Tencent response missing requested symbol')
        quote = data.get('qt', {}).get(symbol)
        if quote and len(quote) > 2 and quote[2] != item['code']:
            raise ValueError('Tencent quote code mismatch')
        if quote and len(quote) > 2 and quote[1] != item['name']:
            raise ValueError(f'Tencent index name mismatch: {quote[1]}')
        candles = data.get('day')
        if candles is None:
            raise ValueError('Unadjusted daily index candles missing; refusing adjusted stock data')
        for candle in candles:
            date, o, c, h, lo = candle[:5]
            # The provider can return observations preceding the requested start.
            if not window['start'] <= date <= window['end']:
                continue
            if date == now.date().isoformat() and now.time() < dt.time(15, 10):
                continue
            o, close, h, lo = map(float, (o, c, h, lo))
            if not all(math.isfinite(v) and v > 0 for v in (o, close, h, lo)) or lo > min(o, close) or h < max(o, close):
                raise ValueError('invalid Tencent OHLC')
            row = {'date': date, 'code': item['code'], 'name': item['name'], 'close': str(c)}
            if date in rows and rows[date] != row:
                raise ValueError('conflicting overlapping Tencent observations')
            rows[date] = row
    return validate_rows([rows[d] for d in sorted(rows)])


def fetch_tencent(item):
    symbol = tencent_symbol(item)
    windows = []
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date()
    for year in range(2005, today.year + 1, 2):
        start = f'{year}-01-01'
        end = min(f'{year+1}-12-31', today.isoformat())
        url = ('https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?' +
               urllib.parse.urlencode({'param': f'{symbol},day,{start},{end},640,qfq'}))
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read(5_000_000)
        windows.append({'start': start, 'end': end, 'url': url,
                        'response_text': raw.decode('utf-8'), 'sha256': hashlib.sha256(raw).hexdigest()})
    snapshot = {'symbol': symbol, 'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'windows': windows}
    rows = parse_tencent(snapshot, item)
    if len(rows) < 127:
        raise ValueError('Tencent history too short')
    previous, _ = load_series(item)
    if previous and (len(rows) < len(previous) or rows[-1]['date'] < previous[-1]['date']):
        raise ValueError('Tencent would regress existing coverage')
    write_json(DATA / 'sources' / 'tencent' / (item['code'] + '.json'), snapshot)
    return len(rows)


def import_release(archive_path, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['release_tag'] != RELEASE:
        raise ValueError('release tag mismatch')
    if archive_path.stat().st_size != manifest['archive_size_bytes']:
        raise ValueError('archive size mismatch')
    if 'sha256:' + sha256(archive_path) != manifest['archive_sha256']:
        raise ValueError('archive digest mismatch')
    symbols = {'sh000300', 'sh000905', 'sh000906', 'sh000852', 'sh000985'}
    selected = {}
    # Read only explicitly selected regular files. No arbitrary extraction or execution.
    with tarfile.open(archive_path, 'r|gz') as tar:
        for member in tar:
            parts = member.name.split('/')
            wanted = member.name == 'qlib_bin/calendars/day.txt' or (
                len(parts) == 4 and parts[:2] == ['qlib_bin', 'features'] and
                parts[2] in symbols and parts[3].endswith('.day.bin'))
            if wanted:
                if not member.isfile() or member.size > 10_000_000:
                    raise ValueError('unsafe selected member')
                name = '/'.join(parts[1:])
                if name in selected:
                    raise ValueError('duplicate selected member')
                selected[name] = tar.extractfile(member).read()
    if 'calendars/day.txt' not in selected:
        raise ValueError('missing observed calendar')
    folder = DATA / 'sources' / 'investment_data' / RELEASE
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / 'selected.tmp'
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        for name, content in sorted(selected.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            out.writestr(info, content)
    temp.replace(folder / 'selected.zip')
    write_json(folder / 'manifest.json', manifest)
    write_json(folder / 'provenance.json', {
        'archive_url': f'{SOURCE}/releases/download/{RELEASE}/qlib_bin.tar.gz',
        'manifest_url': f'{SOURCE}/releases/download/{RELEASE}/qlib_bin.manifest.json',
        'selected_sha256': sha256(folder / 'selected.zip'),
        'selection': sorted(selected),
        'notes': 'Original bytes, selected from checksum-verified archive; full archive retained outside Git.'})


def load_series(item):
    if item.get('community_name'):
        path = DATA / 'sources' / 'shenwan_community' / 'sw_industry_daily.csv'
        if not path.exists():
            return [], None
        provenance = json.loads((path.parent / 'provenance.json').read_text(encoding='utf-8'))
        if sha256(path) != provenance['sha256']:
            raise ValueError('community source digest mismatch')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = [{'date': r['日期'], 'code': item['code'], 'name': item['name'], 'close': r['收盘']}
                    for r in csv.DictReader(stream) if r['代码'] == item['code'] and r['行业名称'] == item['community_name']]
        rows.sort(key=lambda r: r['date'])
        return validate_rows(rows), {'provider': 'shenwan_community', 'url': provenance['url'],
                                    'raw_path': path.relative_to(ROOT).as_posix(), 'raw_sha256': sha256(path),
                                    'price_basis': 'community_reported_price_index_points',
                                    'precision': 'source decimal string; NOT independently verified'}
    path = DATA / 'sources' / 'eastmoney' / (item['code'] + '.json')
    if path.exists():
        rows = parse_eastmoney(json.loads(path.read_text(encoding='utf-8')), item)
        source = {'provider': 'eastmoney', 'url': eastmoney_url(item),
                  'raw_path': path.relative_to(ROOT).as_posix(), 'raw_sha256': sha256(path),
                  'price_basis': 'vendor_price_index_points', 'precision': 'source decimal string'}
        return rows, source
    path = DATA / 'sources' / 'tencent' / (item['code'] + '.json')
    if path.exists():
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        rows = parse_tencent(snapshot, item)
        source = {'provider': 'tencent', 'url': snapshot['windows'][-1]['url'],
                  'raw_path': path.relative_to(ROOT).as_posix(), 'raw_sha256': sha256(path),
                  'retrieved_at': snapshot['retrieved_at'],
                  'price_basis': 'vendor_unadjusted_daily_index_points',
                  'precision': 'source decimal string; independent verification incomplete'}
        return rows, source
    folder = DATA / 'sources' / 'investment_data' / RELEASE
    archive = folder / 'selected.zip'
    if archive.exists():
        provenance = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
        if sha256(archive) != provenance['selected_sha256']:
            raise ValueError('selected archive digest mismatch')
        symbol = 'sh' + item['code']
        with zipfile.ZipFile(archive) as z:
            if f'features/{symbol}/adjclose.day.bin' in z.namelist():
                source = {'provider': 'investment_data', 'url': f'{SOURCE}/releases/tag/{RELEASE}',
                          'raw_path': archive.relative_to(ROOT).as_posix(), 'raw_sha256': sha256(archive),
                          'price_basis': 'upstream_adjclose_index_points',
                          'precision': 'float32 rounded to 2 decimals; independent verification incomplete'}
                return qlib_rows(z, symbol, item), source
    return [], None


def build():
    catalog, all_rows = [], []
    calendar = set()
    upstream = DATA / 'sources' / 'investment_data' / RELEASE / 'selected.zip'
    if upstream.exists():
        with zipfile.ZipFile(upstream) as z:
            calendar = set(z.read('calendars/day.txt').decode().splitlines())
    calendar_end = max(calendar) if calendar else None
    for item in registry():
        rows, source = load_series(item)
        folder = DATA / 'indices' / item['category']
        folder.mkdir(parents=True, exist_ok=True)
        meta = dict(item, status='downloaded' if rows else 'pending', rows=len(rows),
                    start=rows[0]['date'] if rows else None, end=rows[-1]['date'] if rows else None,
                    source=source, history_semantics='May include pre-launch backfill; publication dates not yet audited')
        if rows:
            csv_path = folder / (item['code'] + '.csv')
            buffer = io.StringIO(newline='')
            writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
            csv_path.write_text(buffer.getvalue(), encoding='utf-8-sig', newline='')
            meta.update(csv=csv_path.relative_to(ROOT).as_posix(), csv_sha256=sha256(csv_path))
            all_rows.extend(rows)
            dates = {r['date'] for r in rows}
            missing = sorted(d for d in calendar if meta['start'] <= d <= meta['end'] and d not in dates)
            trailing = sorted(d for d in calendar if d > meta['end'])
            meta['quality'] = {'reference_calendar_end': calendar_end,
                               'missing_observations_within_range': len(missing),
                               'missing_dates': missing,
                               'reference_trading_days_since_last_observation': len(trailing),
                               'independently_verified': False,
                               'warnings': ['Index launch/backfill dates not audited; not a point-in-time dataset.']}
            if missing:
                meta['quality']['warnings'].append('Internal gaps exist; do not fill or treat adjacent rows as adjacent trading days.')
            if trailing:
                meta['quality']['warnings'].append('History ends before the reference calendar; not current market data.')
            if source['provider'] == 'shenwan_community':
                meta['quality']['warnings'].append('Community historical sample only; values and historical classification changes not independently verified.')
        else:
            meta['blocker'] = 'No usable history downloaded: upstream HTTP 503; fallback requires network settings.'
        write_json(folder / (item['code'] + '.meta.json'), meta)
        catalog.append(meta)
    write_json(DATA / 'catalog.json', {'schema_version': 1, 'series': catalog})
    rebuild_sqlite(catalog, all_rows)
    directory_readmes(catalog)
    print('\n'.join(f"{m['category']}/{m['code']} {m['name']}: {m['status']} {m['rows']} {m['start']} ~ {m['end']}" for m in catalog))


def directory_readmes(catalog):
    for category in dict.fromkeys(m['category'] for m in catalog):
        lines = [f'# {category}历史数据', '',
                 '不同代码是不同指数，禁止拼接为一条序列。申万社区样本与中证指数分开保存。', '',
                 '| 指数 | 代码 | 记录数 | 起止日期 | 来源 | 区间内缺日 |',
                 '| --- | --- | ---: | --- | --- | ---: |']
        for m in catalog:
            if m['category'] != category:
                continue
            link = f"[{m['name']}]({m['code']}.csv)" if m['rows'] else m['name'] + '（待补齐）'
            lines.append(f"| {link} | {m['code']} | {m['rows']} | {m['start']} ~ {m['end']} | "
                         f"{m['source']['provider'] if m['source'] else '无'} | "
                         f"{m.get('quality',{}).get('missing_observations_within_range','—')} |")
        lines += ['', '来源、SHA-256、口径和缺失日期详见同名 `.meta.json`；总说明见 [数据说明](../../README.md)。', '']
        (DATA / 'indices' / category / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def rebuild_sqlite(catalog, rows):
    path = DATA / 'market.sqlite3'
    temp = DATA / 'market.tmp.sqlite3'
    temp.unlink(missing_ok=True)
    with sqlite3.connect(temp) as conn:
        conn.executescript('''
        CREATE TABLE series(code TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
          status TEXT NOT NULL, start_date TEXT, end_date TEXT, row_count INTEGER NOT NULL,
          metadata_json TEXT NOT NULL);
        CREATE TABLE prices(code TEXT NOT NULL REFERENCES series(code), date TEXT NOT NULL,
          close REAL NOT NULL CHECK(close>0), PRIMARY KEY(code,date)) WITHOUT ROWID;
        ''')
        conn.executemany('INSERT INTO series VALUES(?,?,?,?,?,?,?,?)', [
            (m['code'],m['name'],m['category'],m['status'],m['start'],m['end'],m['rows'],
             json.dumps(m,ensure_ascii=False,sort_keys=True)) for m in catalog])
        conn.executemany('INSERT INTO prices VALUES(?,?,?)',
                         [(r['code'],r['date'],float(r['close'])) for r in rows])
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    if path.exists() and sha256(path) == sha256(temp):
        temp.unlink()
    else:
        temp.replace(path)


def fetch(codes=None, provider='eastmoney'):
    failed = []
    if provider == 'tencent':
        selected = [m for m in registry() if m.get('secid') and (not codes or m['code'] in codes)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fetch_tencent, m): m for m in selected}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    print(f"OK {item['name']} {future.result()}", flush=True)
                except Exception as exc:
                    print(f"FAILED {item['name']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    failed.append(item['code'])
        build()
        return 1 if failed else 0
    for item in registry():
        if codes and item['code'] not in codes:
            continue
        if not item.get('secid'):
            continue  # Do not silently replace Shenwan histories with CSI or ETF histories.
        try:
            request = urllib.request.Request(eastmoney_url(item), headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=25) as response:
                raw = response.read(10_000_000)
            payload = json.loads(raw)
            rows = parse_eastmoney(payload, item)
            previous, _ = load_series(item)
            if len(rows) < 127 or (previous and (len(rows) < len(previous) or rows[-1]['date'] < previous[-1]['date'])):
                raise ValueError('history is too short or regresses existing coverage')
            path = DATA / 'sources' / 'eastmoney' / (item['code'] + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            write_json(path.with_suffix('.provenance.json'), {
                'url': eastmoney_url(item), 'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'sha256': sha256(path)})
            print(f"OK {item['name']} {len(rows)}", flush=True)
        except Exception as exc:
            print(f"FAILED {item['name']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            failed.append(item['code'])
    build()
    return 1 if failed else 0


def audit():
    catalog = json.loads((DATA / 'catalog.json').read_text(encoding='utf-8'))['series']
    with sqlite3.connect(f'file:{DATA / "market.sqlite3"}?mode=ro', uri=True) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
        assert conn.execute('SELECT count(*) FROM series').fetchone()[0] == len(catalog)
        assert conn.execute('SELECT count(*) FROM prices').fetchone()[0] == sum(m['rows'] for m in catalog)
        for meta in catalog:
            stored_meta = conn.execute('SELECT metadata_json FROM series WHERE code=?', (meta['code'],)).fetchone()
            assert stored_meta and json.loads(stored_meta[0]) == meta
            if meta['status'] != 'downloaded':
                assert meta['rows'] == 0
                assert conn.execute('SELECT count(*) FROM prices WHERE code=?',(meta['code'],)).fetchone()[0] == 0
                continue
            path = ROOT / meta['csv']
            assert sha256(path) == meta['csv_sha256']
            with path.open(encoding='utf-8-sig',newline='') as stream:
                rows = validate_rows(list(csv.DictReader(stream)))
            assert len(rows) == meta['rows']
            stored = conn.execute('SELECT date, close FROM prices WHERE code=? ORDER BY date', (meta['code'],)).fetchall()
            assert stored == [(r['date'],float(r['close'])) for r in rows]
            assert all(r['code'] == meta['code'] and r['name'] == meta['name'] for r in rows)
            assert rows[0]['date'] == meta['start'] and rows[-1]['date'] == meta['end']
            raw = ROOT / meta['source']['raw_path']
            assert sha256(raw) == meta['source']['raw_sha256']
            reproduced, _ = load_series(meta)
            assert rows == reproduced
    print(f"PASS: {sum(m['rows'] for m in catalog)} rows; "
          f"{sum(m['status']=='downloaded' for m in catalog)} downloaded; "
          f"{sum(m['status']=='pending' for m in catalog)} pending.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'fetch', 'audit', 'import-release'])
    parser.add_argument('--codes', nargs='+')
    parser.add_argument('--provider', choices=['eastmoney', 'tencent'], default='eastmoney')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.command == 'import-release':
        if not args.archive or not args.manifest:
            parser.error('--archive and --manifest required')
        import_release(args.archive, args.manifest)
        build()
    elif args.command == 'fetch':
        known = {m['code'] for m in registry()}
        if args.codes and set(args.codes) - known:
            parser.error('unknown index code')
        return fetch(args.codes, args.provider)
    elif args.command == 'build':
        build()
    else:
        audit()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
