"""Validated immutable OHLCV seed and separate daily additions."""
import hashlib
import sqlite3
import threading
import numpy as np
import pandas as pd
from daily.ohlcv_feed import fetch_bars
from daily.feeds import fetch_recent


class OHLCVStore:
    def __init__(self,root,codes,expected_sha,store):
        self.codes=codes;self.lock=threading.RLock();self.pending={};self.errors={};self.store=store
        database=root/'data/enhanced/market_ohlcv.sqlite3'
        if hashlib.sha256(database.read_bytes()).hexdigest()!=expected_sha:raise ValueError('Frozen OHLCV seed changed')
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
            frame=pd.read_sql_query('SELECT * FROM bars ORDER BY code,date',db)
        self.bars={c:f.drop(columns='code').reset_index(drop=True) for c,f in frame.groupby('code') if c in codes}
        store.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(store) as db:
            db.execute('CREATE TABLE IF NOT EXISTS bars(code TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,PRIMARY KEY(code,date))')
            additions=pd.read_sql_query('SELECT * FROM bars ORDER BY code,date',db)
        for code in codes:
            extra=additions[(additions.code==code)&(additions.date>self.bars[code].date.max())].drop(columns='code')
            self._validate(extra)
            if not extra.empty:self.bars[code]=pd.concat([self.bars[code],extra],ignore_index=True)

    @staticmethod
    def _validate(frame):
        if frame.empty:return
        fields=frame[['open','high','low','close','volume']].to_numpy(dtype=float)
        if not np.isfinite(fields).all() or (fields<=0).any():raise ValueError('Invalid OHLCV values')
        if (frame.high<frame[['open','close','low']].max(axis=1)).any() or (frame.low>frame[['open','close','high']].min(axis=1)).any():raise ValueError('Invalid candle range')
        if frame.date.duplicated().any():raise ValueError('Duplicate OHLCV date')

    def fetch(self,item,last_date):
        try:
            bars=fetch_bars(item,last_date);self._validate(pd.DataFrame(bars))
            if not bars:raise ValueError('No completed OHLCV candles')
            with self.lock:self.pending[item['code']]=bars;self.errors.pop(item['code'],None)
            return {r['date']:r['close'] for r in bars},'腾讯'
        except Exception as exc:
            with self.lock:self.pending.pop(item['code'],None);self.errors[item['code']]='量价更新未完成：'+str(exc)
            return fetch_recent(item,last_date)

    def commit(self,accepted):
        fields=['date','open','high','low','close','volume']
        with self.lock,sqlite3.connect(self.store) as db:
            for code,prices in accepted.items():
                incoming=self.pending.pop(code,[])
                if not incoming:continue
                previous=self.bars[code].set_index('date')
                overlap=[r for r in incoming if r['date'] in previous.index]
                if len(overlap)<5 or any(any(abs(previous.loc[r['date'],field]-r[field])>.020001 for field in ('open','high','low','close')) or abs(previous.loc[r['date'],'volume']/r['volume']-1)>.005 for r in overlap):
                    self.errors[code]='量价重叠校验失败，预测保留最后可用日期';continue
                new=[r for r in incoming if r['date']>self.bars[code].date.max() and r['date'] in prices]
                if any(abs(r['close']-prices[r['date']])>.020001 for r in new):
                    self.errors[code]='量价收盘与行情不一致，预测保留最后可用日期';continue
                db.executemany('INSERT OR IGNORE INTO bars VALUES(?,?,?,?,?,?,?)',[(code,)+tuple(r[f] for f in fields) for r in new])
                if new:self.bars[code]=pd.concat([self.bars[code],pd.DataFrame(new)],ignore_index=True)
