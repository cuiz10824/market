"""Prepare five-year enriched samples on observed exchange sessions, without fitting transforms."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from daily.cycle_features import build_frames, feature_names

ROOT=Path(__file__).resolve().parents[2]


def load_bars(root=ROOT):
    path=root/'data/enhanced/market_ohlcv.sqlite3'
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        frame=pd.read_sql_query('SELECT * FROM bars ORDER BY code,date',db)
        calendar=[r[0] for r in db.execute('SELECT date FROM calendar ORDER BY date')]
    return {code:f.drop(columns='code').reset_index(drop=True) for code,f in frame.groupby('code')},calendar


def samples(bars,codes,calendar):
    features=build_frames(bars,codes,calendar)
    names=feature_names(codes);result={}
    for h in (1,5,21,63,126):
        parts=[]
        for code in codes:
            close=bars[code].set_index('date').close.reindex(calendar)
            frame=features[code].copy()
            frame['target_return']=close.shift(-h)/close-1
            # Require every future observation through the target; no skipping gaps.
            continuous=close.notna().astype(int).rolling(h+1).sum().shift(-h)==h+1
            frame['date']=calendar
            frame['label_end_date']=pd.Series(calendar,index=calendar).shift(-h)
            frame['code']=code
            frame=frame.loc[continuous].dropna(subset=list(names)+['target_return','label_end_date'])
            parts.append(frame)
        result[h]=pd.concat(parts).sort_values(['date','code']).reset_index(drop=True)
        if not np.isfinite(result[h][list(names)+['target_return']].to_numpy()).all():raise ValueError('Nonfinite enriched samples')
    return result,features
