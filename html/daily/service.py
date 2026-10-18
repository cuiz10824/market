"""Merge daily observations with immutable training data, then evaluate regressors."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import tempfile
import zipfile

from daily.features import features_at
from daily.feeds import fetch_recent, SHANGHAI
from daily.regression import RegressionPredictor

ROOT=Path(__file__).resolve().parents[2]
HORIZONS=(1,5,21,63,126)
MODEL_NAMES={'zero':'零变化基准','mean':'历史平均收益','ridge':'岭回归','lightgbm':'LightGBM回归'}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def merge_checked(old,new,calendar):
    common=set(old)&set(new)
    if len(common)<5:
        raise ValueError('与本地历史重叠不足5日，拒绝拼接')
    if any(abs(old[d]-new[d])>.020001 for d in common):
        raise ValueError('与本地历史收盘不一致，拒绝拼接，请核验行情口径')
    if max(new)<max(old):
        raise ValueError('行情源落后于本地历史')
    unknown=set(new)-set(calendar)
    if unknown:
        raise ValueError('新增行情未在沪深300交易日历中得到确认')
    result=dict(old)
    result.update({day:value for day,value in new.items() if day>max(old)})
    last=max(result)
    dates=[d for d in calendar if d<=last][-253:]
    if len(dates)<253 or any(day not in result for day in dates):
        raise ValueError('最近252个交易日存在缺口，拒绝生成预测')
    return result


class DailyService:
    def __init__(self,root=ROOT,store=None,fetcher=fetch_recent):
        self.root=Path(root)
        self.store=Path(store) if store else self.root/'data/daily/market.sqlite3'
        self.fetcher=fetcher
        self.lock=threading.RLock()
        self.refreshing=False
        self.items=[i for i in load_json(self.root/'data/registry.json') if i.get('secid')]
        pointer=load_json(self.root/'code/models/latest.json')
        index=load_json(self.root/pointer['index'])
        manifest=load_json(self.root/pointer['verification'])['all_saved_file_sha256']
        self.run_id=pointer['run_id']
        self.models={}
        for entry in index['models']:
            model=RegressionPredictor(self.root/entry['latest'],manifest[entry['latest']])
            if model.bundle['code']!=entry['code'] or model.bundle['horizon']!=entry['horizon']:
                raise ValueError('模型索引不匹配')
            self.models[(entry['code'],entry['horizon'])]=model
        assert set(self.models)=={(i['code'],h) for i in self.items for h in HORIZONS if h!=1}
        with zipfile.ZipFile(self.root/'data/sources/investment_data/2026-10-17/selected.zip') as z:
            self.calendar=z.read('calendars/day.txt').decode().splitlines()
        baseline=self.root/'data/market.sqlite3'
        expected={m.bundle['metadata']['database_sha256'] for m in self.models.values()}
        if expected!={hashlib.sha256(baseline.read_bytes()).hexdigest()}:
            raise ValueError('原训练数据库已变化；请恢复冻结数据库，新增行情应写入data/daily')
        database_uri=(self.root/'data/market.sqlite3').resolve().as_uri()+'?mode=ro'
        with sqlite3.connect(database_uri,uri=True) as db:
            self.prices={i['code']:dict(db.execute('SELECT date,close FROM prices WHERE code=? ORDER BY date',(i['code'],))) for i in self.items}
        self.store.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.store) as db:
            db.executescript('CREATE TABLE IF NOT EXISTS prices(code TEXT,date TEXT,close REAL,provider TEXT,PRIMARY KEY(code,date)); CREATE TABLE IF NOT EXISTS calendar(date TEXT PRIMARY KEY); CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);')
            self.calendar=sorted(set(self.calendar)|{r[0] for r in db.execute('SELECT date FROM calendar')})
            for code,day,close in db.execute('SELECT code,date,close FROM prices ORDER BY code,date'):
                if code in self.prices and day>max(self.prices[code]):
                    self.prices[code][day]=close
            stored=db.execute("SELECT value FROM metadata WHERE key='last_refresh'").fetchone()
        self.last_refresh=json.loads(stored[0]) if stored else {'attempted_at':None,'errors':{},'providers':{}}
        self.payload=self.build_payload()

    def build_payload(self):
        now=dt.datetime.now(SHANGHAI)
        rows=[]
        for item in self.items:
            code=item['code'];prices=self.prices[code];last=max(prices)
            dates=[d for d in self.calendar if d<=last]
            tail=dates[-253:]
            valid=len(tail)==253 and all(d in prices for d in tail)
            feature=features_at([prices[d] for d in tail],252) if valid else None
            actual={str(h):(prices[last]/prices[dates[-1-h]]-1)*100
                    if len(dates)>h and dates[-1-h] in prices else None for h in HORIZONS}
            predictions={'1':None};kinds={}
            for h in HORIZONS[1:]:
                model=self.models[(code,h)]
                # Never run latest refits against observations preceding their training cutoff.
                eligible=feature is not None and last>=model.bundle['metadata']['fit_label_end']
                predictions[str(h)]=model.predict(feature)*100 if eligible else None
                kinds[str(h)]=MODEL_NAMES[model.spec['kind']]
            error=self.last_refresh.get('errors',{}).get(code)
            rows.append({'code':code,'name':item['name'],'category':item['category'],'note':item['note'],
                         'as_of':last,'close':prices[last],'actual':actual,'forecast':predictions,
                         'models':kinds,'data_status':'更新失败，使用本地历史' if error else '本地数据库及已验证增量',
                         'update_error':error,'stale':(now.date()-dt.date.fromisoformat(last)).days>4,
                         'source':self.last_refresh.get('providers',{}).get(code,'本地历史数据库')})
        return {'schema_version':1,'generated_at':now.isoformat(),'model_run':self.run_id,
                'unit':'percent','horizons':list(HORIZONS),'rows':rows,'research_only':True,
                'forecast_basis':'已收盘价格；不含分红及费用；现有训练模型仅支持5/21/63/126交易日',
                'last_refresh':self.last_refresh}

    def dashboard(self):
        with self.lock:
            return dict(self.payload,refreshing=self.refreshing)

    def refresh(self):
        """One caller at a time; partial failures preserve that index's previous history."""
        with self.lock:
            if self.refreshing:
                return False
            self.refreshing=True
        try:
            fetched={};errors={};providers={}
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures={pool.submit(self.fetcher,item,max(self.prices[item['code']])):item for item in self.items}
                for future in as_completed(futures):
                    code=futures[future]['code']
                    try:
                        fetched[code],providers[code]=future.result()
                    except Exception as exc:
                        errors[code]=str(exc)
            calendar=list(self.calendar)
            # Validate anchor overlap before treating its new dates as exchange sessions.
            if '000300' in fetched:
                candidate=sorted(set(calendar)|set(fetched['000300']))
                try:
                    merge_checked(self.prices['000300'],fetched['000300'],candidate)
                    calendar=candidate
                except ValueError as exc:
                    errors['000300']=str(exc)
                    del fetched['000300']
            accepted={}
            for code,new in fetched.items():
                try:
                    accepted[code]=merge_checked(self.prices[code],new,calendar)
                except ValueError as exc:
                    errors[code]=str(exc)
            info={'attempted_at':dt.datetime.now(SHANGHAI).isoformat(),'errors':errors,
                  'providers':{c:p for c,p in providers.items() if c in accepted},
                  'successful_indices':len(accepted),'total_indices':len(self.items)}
            with self.lock:
                with sqlite3.connect(self.store) as db:
                    db.executemany('INSERT OR IGNORE INTO calendar VALUES(?)',[(d,) for d in calendar if d not in self.calendar])
                    for code,prices in accepted.items():
                        new=[(code,d,c,providers[code]) for d,c in prices.items() if d not in self.prices[code]]
                        db.executemany('INSERT INTO prices VALUES(?,?,?,?)',new)
                    db.execute("INSERT OR REPLACE INTO metadata VALUES('last_refresh',?)",(json.dumps(info,ensure_ascii=False),))
                self.prices.update(accepted)
                self.calendar=calendar
                self.last_refresh=info
                self.payload=self.build_payload()
            return True
        finally:
            with self.lock:
                self.refreshing=False

    def export_snapshot(self):
        payload=json.dumps(self.dashboard(),ensure_ascii=False,allow_nan=False)
        path=self.root/'html/snapshot.js'
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix='snapshot-',suffix='.tmp',delete=False) as stream:
            temp=Path(stream.name)
            stream.write('window.MARKET_SNAPSHOT = '+payload+';\n')
        try:
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    service=DailyService()
    if args.refresh:service.refresh()
    service.export_snapshot()
    print(json.dumps({'rows':len(service.payload['rows']),'last_refresh':service.last_refresh},ensure_ascii=False))
