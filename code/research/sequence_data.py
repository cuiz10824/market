"""Past-only five-session OHLCV blocks, shared origins and masked direct-return labels."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
CHANNELS=('log_close_vs_origin_close','opening_gap','high_low_range','intrabar_return','relative_daily_volume','block_return')
SCALES=np.array([1.,.02,.05,.02,1.,.03],dtype=np.float64)


def pack(raw,ends,lookback):
    """Each origin reads exactly lookback completed daily bars and their preceding close."""
    width=(lookback+4)//5
    finish=ends[:,None]-5*np.arange(width-1,-1,-1)[None,:]
    begin=np.maximum(finish-4,ends[:,None]-lookback+1)
    high=np.full(finish.shape,-np.inf);low=np.full(finish.shape,np.inf);volume=np.zeros(finish.shape)
    for lag in range(5):
        ix=finish-lag;valid=ix>=begin
        high=np.maximum(high,np.where(valid,raw[ix,1],-np.inf))
        low=np.minimum(low,np.where(valid,raw[ix,2],np.inf))
        volume+=np.where(valid,raw[ix,4],0.)
    opening=raw[begin,0];close=raw[finish,3];prior=raw[begin-1,3]
    cumulative=np.r_[0.,np.cumsum(np.nan_to_num(raw[:,4]))]
    mean_volume=(cumulative[ends+1]-cumulative[ends-lookback+1])/lookback
    daily_volume=volume/(finish-begin+1)
    x=np.stack([np.log(close/raw[ends,3,None]),np.log(opening/prior),np.log(high/low),
                np.log(close/opening),np.log(daily_volume/mean_volume[:,None]),np.log(close/prior)],axis=-1)
    x=(x/SCALES).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite sequence inputs')
    return x


class SequenceData:
    def __init__(self,p,bars=None,calendar=None):
        self.p=p;self.codes=p['codes'];self.horizons=p['horizons']
        if bars is None:
            with sqlite3.connect((ROOT/'data/enhanced/market_ohlcv.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
                frame=pd.read_sql_query('SELECT * FROM bars ORDER BY code,date',db)
                calendar=[r[0] for r in db.execute('SELECT date FROM calendar ORDER BY date')]
            bars={c:f.drop(columns='code') for c,f in frame.groupby('code')}
        self.calendar=np.asarray(calendar);self.raw={};self.x={};ids=[];ends_all=[];ys=[];known=[]
        chunks={length:[] for length in p['lookbacks']}
        warmup=max(p['lookbacks'])
        for identity,code in enumerate(self.codes):
            raw=bars[code].set_index('date').reindex(calendar)[['open','high','low','close','volume']].to_numpy(dtype=float)
            self.raw[code]=raw
            good=np.isfinite(raw).all(axis=1)&(raw>0).all(axis=1)
            bad=np.r_[0,np.cumsum(~good)]
            candidates=np.arange(warmup,len(calendar))
            ends=candidates[(bad[candidates+1]-bad[candidates-warmup])==0]
            ids.extend([identity]*len(ends));ends_all.extend(ends)
            y=np.zeros((len(ends),len(self.horizons)),dtype=np.float64);mask=np.zeros_like(y,dtype=bool)
            invalid=np.r_[0,np.cumsum(~np.isfinite(raw[:,3]))]
            for j,h in enumerate(self.horizons):
                target=ends+h;bounded=target<len(calendar)
                eligible=np.where(bounded)[0];future=target[bounded]
                complete=(invalid[future+1]-invalid[ends[bounded]])==0
                ix=eligible[complete];final=target[ix]
                mask[ix,j]=True;y[ix,j]=raw[final,3]/raw[ends[ix],3]-1
            ys.append(y);known.append(mask)
            for length in p['lookbacks']:chunks[length].append(pack(raw,ends,length))
        self.ids=np.asarray(ids,dtype=np.int64);self.ends=np.asarray(ends_all,dtype=np.int64)
        order=np.lexsort((self.ids,self.ends))
        self.ids=self.ids[order];self.ends=self.ends[order]
        self.y=np.concatenate(ys)[order];self.known=np.concatenate(known)[order]
        self.dates=self.calendar[self.ends]
        self.label_indices=self.ends[:,None]+np.asarray(self.horizons)[None,:]
        self.target_scales=.01*np.sqrt(np.asarray(self.horizons,dtype=np.float32))
        for length in p['lookbacks']:self.x[length]=np.concatenate(chunks[length])[order]

    def split(self,start,end=None,train=False,stride=1):
        mask=self.known.copy()
        if train:
            limit=int(np.searchsorted(self.calendar,start)) if start else len(self.calendar)
            mask &= self.label_indices<limit
            selected=(self.ends<limit)&mask.any(axis=1)
        else:
            selected=self.dates>=start
            if end:
                mask &= self.label_indices<int(np.searchsorted(self.calendar,end))
                selected &= self.dates<end
            selected &= mask.any(axis=1)
        selected &= self.ends%stride==0
        rows=np.where(selected)[0]
        return rows,mask[rows]
