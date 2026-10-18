"""Fixed public index feeds; only completed, unadjusted daily candles."""
import datetime as dt
import json
import math
import urllib.parse
import urllib.request

SHANGHAI=dt.timezone(dt.timedelta(hours=8))


def completed(candles, now=None):
    now=now or dt.datetime.now(SHANGHAI)
    rows={}
    previous=''
    for candle in candles:
        if len(candle)<5:
            raise ValueError('行情缺少开高低收')
        day=dt.date.fromisoformat(candle[0])
        if day.isoformat()!=candle[0] or day>now.date() or day.weekday()>4:
            raise ValueError('行情日期无效')
        if candle[0]<=previous:
            raise ValueError('行情日期重复或顺序错误')
        previous=candle[0]
        o,c,h,lo=map(float,candle[1:5])
        if not all(math.isfinite(v) and v>0 for v in (o,c,h,lo)) or lo>min(o,c) or h<max(o,c) or lo>h:
            raise ValueError('行情价格无效')
        if day==now.date() and now.time()<dt.time(15,10):
            continue
        rows[day.isoformat()]=c
    if not rows:
        raise ValueError('行情未包含已收盘记录')
    return rows


def parse_tencent(payload,item,now=None):
    symbol=('sz' if item['secid'].startswith('0.') else 'sh')+item['code']
    data=payload.get('data',{}).get(symbol)
    if payload.get('code')!=0 or not isinstance(data,dict) or 'day' not in data:
        raise ValueError('腾讯未返回指定指数的不复权日线')
    quote=data.get('qt',{}).get(symbol)
    if not quote or len(quote)<3 or quote[2]!=item['code'] or quote[1]!=item['name']:
        raise ValueError('腾讯指数代码或名称不匹配')
    return completed(data['day'],now)


def parse_eastmoney(payload,item,now=None):
    data=payload.get('data')
    if payload.get('rc')!=0 or not data or data.get('code')!=item['code'] or data.get('name')!=item['name']:
        raise ValueError('东方财富指数代码或名称不匹配')
    return completed([line.split(',') for line in data.get('klines',[])],now)


def fetch_recent(item,last_date):
    today=dt.datetime.now(SHANGHAI).date().isoformat()
    start=(dt.date.fromisoformat(last_date)-dt.timedelta(days=45)).isoformat()
    symbol=('sz' if item['secid'].startswith('0.') else 'sh')+item['code']
    endpoints=[('腾讯', 'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?'+
                urllib.parse.urlencode({'param':f'{symbol},day,{start},{today},640,qfq'}),parse_tencent),
               ('东方财富','https://push2his.eastmoney.com/api/qt/stock/kline/get?'+urllib.parse.urlencode({
                   'secid':item['secid'],'klt':101,'fqt':0,'lmt':640,'beg':start.replace('-',''),
                   'end':today.replace('-',''),'fields1':'f1,f2,f3,f4,f5,f6',
                   'fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'}),parse_eastmoney)]
    errors=[]
    for provider,url,parser in endpoints:
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(request,timeout=10) as response:
                raw=response.read(2_000_001)
            if len(raw)>2_000_000:
                raise ValueError('行情响应过大')
            return parser(json.loads(raw),item),provider
        except Exception as exc:
            errors.append(f'{provider}: {type(exc).__name__}: {exc}')
    raise ValueError('；'.join(errors))
