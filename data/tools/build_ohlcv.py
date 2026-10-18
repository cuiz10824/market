"""Recover validated OHLCV from preserved Tencent responses; never invent missing bars."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'html'))
from daily.selection import load_selection

FIELDS=('date','open','high','low','close','volume')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bar(raw):
    day=dt.date.fromisoformat(raw[0])
    if day.isoformat()!=raw[0] or day.weekday()>4:raise ValueError('Invalid session date')
    o,c,h,lo,v=map(float,raw[1:6])
    if not all(math.isfinite(x) and x>0 for x in (o,c,h,lo,v)):raise ValueError('Invalid OHLCV')
    if lo>min(o,c) or h<max(o,c) or lo>h:raise ValueError('Inconsistent OHLC')
    return dict(date=raw[0],open=o,high=h,low=lo,close=c,volume=v)


def build():
    output=ROOT/'data/enhanced';output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ROOT/'data/sources/investment_data/2026-10-17/selected.zip') as z:
        calendar=z.read('calendars/day.txt').decode().splitlines()
    with sqlite3.connect((ROOT/'data/market.sqlite3').as_uri()+'?mode=ro',uri=True) as baseline:
        records=[];series=[];sources={}
        for item in load_selection(ROOT):
            code=item['code'];path=ROOT/'data/sources/tencent'/(code+'.json')
            snapshot=json.loads(path.read_text(encoding='utf-8'));sources[str(path.relative_to(ROOT))]=sha(path)
            symbol=('sz' if item['secid'].startswith('0.') else 'sh')+code
            if snapshot['symbol']!=symbol:raise ValueError('Wrong snapshot symbol')
            bars={}
            for window in snapshot['windows']:
                if hashlib.sha256(window['response_text'].encode()).hexdigest()!=window['sha256']:raise ValueError('Raw response changed')
                payload=json.loads(window['response_text'])
                if payload['code']!=0 or symbol not in payload['data']:raise ValueError('Wrong response identity')
                data=payload['data'][symbol]
                quote=data.get('qt',{}).get(symbol)
                if quote and (quote[2]!=code or quote[1]!=item['name']):raise ValueError('Index name/code mismatch')
                if 'day' not in data:raise ValueError('Unadjusted daily candles required')
                for raw in data['day']:
                    if not window['start']<=raw[0]<=window['end']:continue
                    bar=validate_bar(raw)
                    if bar['date'] in bars and bars[bar['date']]!=bar:raise ValueError('Conflicting overlaps')
                    bars[bar['date']]=bar
            prices=dict(baseline.execute('SELECT date,close FROM prices WHERE code=?',(code,)))
            if set(bars)!=set(prices):raise ValueError('OHLCV coverage differs from frozen closes')
            if any(abs(bars[d]['close']-prices[d])>.020001 for d in prices):raise ValueError('Closes do not match training snapshot')
            observed=[d for d in calendar if min(bars)<=d<=max(bars)]
            if set(observed)!=set(bars):raise ValueError('Calendar gaps')
            for day in sorted(bars):records.append((code,)+tuple(bars[day][f] for f in FIELDS))
            folder=output/item['category'];folder.mkdir(exist_ok=True)
            with (folder/(code+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(bars[d] for d in sorted(bars))
            series.append({'code':code,'name':item['name'],'rows':len(bars),'start':min(bars),'end':max(bars),
                           'calendar_gaps':0,'close_mismatches':0,'zero_volumes':0})
    database=output/'market_ohlcv.sqlite3';temp=output/'market_ohlcv.tmp'
    temp.unlink(missing_ok=True)
    with sqlite3.connect(temp) as db:
        db.execute('CREATE TABLE bars(code TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,PRIMARY KEY(code,date))')
        db.executemany('INSERT INTO bars VALUES(?,?,?,?,?,?,?)',records)
        db.execute('CREATE TABLE calendar(date TEXT PRIMARY KEY)')
        db.executemany('INSERT INTO calendar VALUES(?)',[(d,) for d in calendar])
    temp.replace(database)
    manifest={'schema_version':1,'database_sha256':sha(database),'baseline_database_sha256':sha(ROOT/'data/market.sqlite3'),
              'source_sha256':sources,'series':series,'rows':len(records),
              'volume_unit':'Tencent raw provider units; only within-series relative-volume features are used',
              'amount':'Not imported: ambiguous provider field units; never treated as verified turnover amount',
              'limitations':['A500 includes retrospective pre-launch history','No point-in-time valuations or macro data imported']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
