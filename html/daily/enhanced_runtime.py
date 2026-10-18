"""Apply validation-approved upgrades; preserve legacy output when rich data are unavailable."""
import hashlib
import json
import sqlite3
import threading

import numpy as np
import pandas as pd

from daily.enhanced_predictor import EnhancedPredictor
from daily.ohlcv_feed import fetch_bars
from daily.feeds import fetch_recent
from daily.rich_features import build_frames,feature_names


def read(path):return json.loads(path.read_text(encoding='utf-8'))


class EnhancedRuntime:
    def __init__(self,root,codes,store=None,pointer_name='latest_enhanced.json'):
        self.root=root;self.codes=codes;self.lock=threading.RLock()
        self.lookback=252
        self.build_frames=build_frames;self.feature_names=feature_names
        if pointer_name=='latest_five_year.json':
            from daily.cycle_features import build_frames as cycle_frames,feature_names as cycle_names
            self.lookback=1260;self.build_frames=cycle_frames;self.feature_names=cycle_names
        elif pointer_name!='latest_enhanced.json':raise ValueError('Unknown feature profile')
        pointer=read(root/'code/models'/pointer_name);index=read(root/pointer['index'])
        checksums=read(root/pointer['verification'])['all_saved_file_sha256']
        self.run_id=pointer['run_id'];self.models={};self.pending={};self.errors={}
        for task in index['tasks']:
            if task['code'] not in codes:continue
            if task['choice']=='legacy':continue
            path=root/task['latest_bundle']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=checksums[task['latest_bundle']]:raise ValueError('Rich bundle checksum differs')
            key=task['latest_bundle']
            if key not in self.models:self.models[key]=EnhancedPredictor(path)
        self.tasks={(t['code'],t['horizon']):t for t in index['tasks'] if t['code'] in codes and t['choice']!='legacy'}
        if not self.tasks:
            self.bars={};return
        expected=self.feature_names(codes)
        for model in self.models.values():
            if tuple(model.bundle['features'])!=expected:raise ValueError('Enhanced feature/order mismatch')
        database=root/'data/enhanced/market_ohlcv.sqlite3'
        if hashlib.sha256(database.read_bytes()).hexdigest()!=index['database_sha256']:raise ValueError('Frozen OHLCV database changed')
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
            frame=pd.read_sql_query('SELECT * FROM bars ORDER BY code,date',db)
        self.bars={c:f.drop(columns='code').reset_index(drop=True) for c,f in frame.groupby('code') if c in codes}
        self.store=store or root/'data/daily/ohlcv.sqlite3';self.store.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.store) as db:
            db.execute('CREATE TABLE IF NOT EXISTS bars(code TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,PRIMARY KEY(code,date))')
            additions=pd.read_sql_query('SELECT * FROM bars ORDER BY code,date',db)
        for code in codes:
            extra=additions[(additions.code==code)&(additions.date>self.bars[code].date.max())].drop(columns='code')
            self.bars[code]=pd.concat([self.bars[code],extra],ignore_index=True)

    def fetch(self,item,last_date):
        try:
            bars=fetch_bars(item,last_date)
            with self.lock:self.pending[item['code']]=bars;self.errors.pop(item['code'],None)
            return {r['date']:r['close'] for r in bars},'腾讯'
        except Exception as exc:
            with self.lock:self.errors[item['code']]='增强量价源不可用：'+str(exc)
            return fetch_recent(item,last_date)

    def commit(self,accepted):
        if not self.tasks:return
        fields=['date','open','high','low','close','volume']
        with self.lock,sqlite3.connect(self.store) as db:
            for code in accepted:
                incoming=self.pending.pop(code,[])
                if not incoming:continue
                previous=self.bars[code].set_index('date')
                overlap=[r for r in incoming if r['date'] in previous.index]
                if len(overlap)<5 or any(abs(previous.loc[r['date'],'close']-r['close'])>.020001 or
                   abs(previous.loc[r['date'],'volume']/r['volume']-1)>.005 for r in overlap):
                    self.errors[code]='量价重叠或成交量单位校验未通过，沿用旧模型';continue
                new=[r for r in incoming if r['date']>self.bars[code].date.max() and r['date'] in accepted[code]]
                db.executemany('INSERT OR IGNORE INTO bars VALUES(?,?,?,?,?,?,?)',[(code,)+tuple(r[f] for f in fields) for r in new])
                if new:self.bars[code]=pd.concat([self.bars[code],pd.DataFrame(new)],ignore_index=True)

    def apply(self,rows,calendar):
        if not self.tasks:return {'run_id':self.run_id,'lookback_sessions':self.lookback,'upgraded_tasks':0,'status':'no_validation_upgrade'}
        frames=self.build_frames(self.bars,self.codes,calendar)
        failed=[];used=0
        for row in rows:
            code=row['code'];date=row['as_of']
            for h in (1,5,21,63,126):
                task=self.tasks.get((code,h))
                if task is None:continue
                model=self.models[task['latest_bundle']]
                feature=frames[code].loc[date]
                if date<model.bundle['metadata']['fit_label_end'] or not np.isfinite(feature.to_numpy()).all():
                    row['enhancement_error']='增强特征未就绪，保留旧模型结果';failed.append(code);continue
                if self.bars[code].date.max()!=date or abs(float(self.bars[code].iloc[-1].close)-row['close'])>.020001:
                    row['enhancement_error']='增强量价数据与最新收盘日期不一致，保留旧模型结果';failed.append(code);continue
                value=float(model.predict([feature.to_dict()])[0])*100
                row['forecast'][str(h)]=value
                family='CatBoost' if task['choice'].startswith('catboost') else 'LightGBM'
                row['models'][str(h)]=family+('五年量价回归' if self.lookback==1260 else '量价回归');used+=1
            if code in self.errors:row['enhancement_error']=self.errors[code]
        return {'run_id':self.run_id,'lookback_sessions':self.lookback,'upgraded_tasks':used,'eligible_upgrades':len(self.tasks),
                'status':'applied' if not failed else 'partial_legacy_fallback','failed_codes':sorted(set(failed))}
