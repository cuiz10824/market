"""Download same-source annual OHLCV with overlap checks and immutable snapshots."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.parse
import urllib.request
import zipfile

import pandas as pd

ROOT=Path(__file__).resolve().parent
SHANGHAI=dt.timezone(dt.timedelta(hours=8))


def sha(data):return hashlib.sha256(data).hexdigest()


def parse(payload, now):
    data=payload.get('data',{}).get('sh000001',{})
    quote=data.get('qt',{}).get('sh000001',[])
    if payload.get('code')!=0 or len(quote)<3 or quote[1:3]!=['上证指数','000001']:
        raise ValueError('Expected SHANGHAI COMPOSITE sh000001, not Shenzhen stock000001')
    result=[];previous=''
    for candle in data.get('day',[]):
        day=dt.date.fromisoformat(candle[0])
        if day.isoformat()<=previous or day>now.date():raise ValueError('Invalid daily date/order')
        previous=day.isoformat()
        if day==now.date() and now.time()<dt.time(15,10):continue
        o,c,h,lo,v=map(float,candle[1:6])
        import math
        if not all(math.isfinite(x) for x in (o,c,h,lo,v)) or min(o,c,h,lo)<=0 or v<0 or lo>min(o,c) or h<max(o,c):
            raise ValueError('Invalid OHLCV candle')
        # Do not discard historical Saturday sessions using today's weekday rules.
        result.append(dict(date=day.isoformat(),open=o,high=h,low=lo,close=c,volume=v))
    if not result:raise ValueError('No completed daily candles')
    return result


def fetch_year(year,now):
    start=f'{year-1}-11-01' if year>1990 else '1990-01-01'
    end=min(f'{year}-12-31',now.date().isoformat())
    url='https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?'+urllib.parse.urlencode(
        {'param':f'sh000001,day,{start},{end},640,qfq'})
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request,timeout=20) as response:raw=response.read(2_000_001)
    if len(raw)>2_000_000:raise ValueError('Response too large')
    rows=parse(json.loads(raw),now)
    if not any(r['date'].startswith(str(year)) for r in rows):raise ValueError(f'No data for{year}')
    return rows,raw,dict(year=year,url=url,response_sha256=sha(raw),requested_start=start,requested_end=end)


def collect():
    now=dt.datetime.now(SHANGHAI)
    fetched={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch_year,y,now):y for y in range(1990,now.year+1)}
        for future in as_completed(futures):
            y=futures[future];fetched[y]=future.result()
            print(f'{y}: {len(fetched[y][0])} rows including overlap',flush=True)
    combined={};sources=[]
    for year in sorted(fetched):
        rows,raw,meta=fetched[year]
        overlap=[r for r in rows if r['date'] in combined]
        if year>1990 and len(overlap)<5:raise ValueError(f'Insufficient cross-window overlap:{year}')
        for row in overlap:
            previous=combined[row['date']]
            if any(abs(previous[f]-row[f])>.020001 for f in ('open','high','low','close')) or abs(previous['volume']-row['volume'])>max(1.,previous['volume']*.00001):
                raise ValueError('Inconsistent overlapping candle:'+row['date'])
        combined.update({r['date']:r for r in rows if r['date'].startswith(str(year))})
        sources.append(dict(meta,overlap_rows=len(overlap),rows_for_year=sum(r['date'].startswith(str(year)) for r in rows)))
    frame=pd.DataFrame([combined[d] for d in sorted(combined)])
    reference=ROOT.parent/'data/sources/investment_data/2026-10-17/selected.zip'
    coverage={'status':'single_source_observed_dates_only'}
    if reference.exists():
        with zipfile.ZipFile(reference) as archive:calendar=archive.read('calendars/day.txt').decode().splitlines()
        relevant=[d for d in calendar if frame.date.min()<=d<=frame.date.max()]
        missing=sorted(set(relevant)-set(frame.date))
        if missing:raise ValueError('Missing observed reference sessions:'+str(missing[:10]))
        coverage=dict(status='all_reference_sessions_present',reference=str(reference.relative_to(ROOT.parent)),
                      reference_sha256=sha(reference.read_bytes()),start=min(relevant),end=max(relevant),sessions=len(relevant),missing=[])
    content=frame.to_csv(index=False,float_format='%.8f',lineterminator='\n').encode()
    dataset_id=now.strftime('%Y%m%d')+'_'+sha(content)[:12]
    folder=ROOT/'data'/dataset_id
    if not folder.exists():
        folder.mkdir(parents=True)
        (folder/'sh000001.csv').write_bytes(content)
        raw_folder=folder/'raw';raw_folder.mkdir()
        for year,(_,raw,_) in fetched.items():
            (raw_folder/f'{year}.json.gz').write_bytes(gzip.compress(raw,mtime=0))
        with sqlite3.connect(folder/'market.sqlite3') as database:
            frame.to_sql('bars',database,index=False)
            database.execute('CREATE UNIQUE INDEX date_idx ON bars(date)')
        manifest=dict(dataset_id=dataset_id,index=dict(symbol='sh000001',code='000001',name='上证指数',exchange='SSE'),
                      first_date=frame.date.min(),last_date=frame.date.max(),rows=len(frame),retrieved_at=now.isoformat(),
                      fields=['date','open','high','low','close','volume'],csv_sha256=sha(content),
                      adjustment='unadjusted index day field; index is not a fund total-return series',
                      volume_unit='Tencent native units, not independently unit-calibrated; relative same-source volume features only',
                      amount='Not supplied: no invented amount column',calendar_coverage=coverage,sources=sources,
                      weekend_dates=frame.loc[pd.to_datetime(frame.date).dt.weekday>4,'date'].tolist(),
                      zero_volume_dates=frame.loc[frame.volume==0,'date'].tolist())
        (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pointer=dict(dataset_id=dataset_id,csv=str((folder/'sh000001.csv').relative_to(ROOT)),
                 manifest=str((folder/'manifest.json').relative_to(ROOT)))
    (ROOT/'data/latest.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(pointer,ensure_ascii=False),flush=True)


if __name__=='__main__':collect()
