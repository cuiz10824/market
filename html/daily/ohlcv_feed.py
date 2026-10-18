"""Tencent-only OHLCV updates, preserving the frozen provider's volume units."""
import datetime as dt
import json
import math
import urllib.parse
import urllib.request

from daily.feeds import SHANGHAI,completed


def fetch_bars(item,last_date):
    symbol=('sz' if item['secid'].startswith('0.') else 'sh')+item['code']
    start=(dt.date.fromisoformat(last_date)-dt.timedelta(days=45)).isoformat()
    end=dt.datetime.now(SHANGHAI).date().isoformat()
    url='https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?'+urllib.parse.urlencode({'param':f'{symbol},day,{start},{end},640,qfq'})
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request,timeout=10) as response:raw=response.read(2_000_001)
    if len(raw)>2_000_000:raise ValueError('OHLCV response too large')
    payload=json.loads(raw);data=payload.get('data',{}).get(symbol)
    if payload.get('code')!=0 or not data or 'day' not in data:raise ValueError('Missing unadjusted index OHLCV')
    quote=data.get('qt',{}).get(symbol)
    if not quote or quote[1]!=item['name'] or quote[2]!=item['code']:raise ValueError('OHLCV identity mismatch')
    candles=[r for r in data['day'] if start<=r[0]<=end]
    closes=completed(candles);bars=[]
    for r in candles:
        if r[0] not in closes:continue
        v=float(r[5])
        if not math.isfinite(v) or v<=0:raise ValueError('OHLCV volume invalid')
        bars.append({'date':r[0],'open':float(r[1]),'high':float(r[3]),'low':float(r[4]),'close':float(r[2]),'volume':v})
    return bars
