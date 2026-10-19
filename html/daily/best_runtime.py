"""Daily inference for the explicitly selected best fitted regressors; no training."""
import hashlib
import json
from pathlib import Path
import numpy as np
from daily.enhanced_predictor import EnhancedPredictor
from daily.regression import RegressionPredictor
from daily.features import one_day_features
from daily.rich_features import build_frames as rich_frames
from daily.cycle_features import build_frames as cycle_frames
from daily.ohlcv_store import OHLCVStore


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class BestRuntime(OHLCVStore):
    def __init__(self,root,codes,store):
        self.root=root
        self.manifest=json.loads((root/'html/daily/model_selection.json').read_text(encoding='utf-8'))
        m=self.manifest
        if m.get('schema_version')!=1 or m['codes']!=codes or m['horizons']!=[1,5,21,63,126]:raise ValueError('Daily selection scope differs')
        if not m['exclude_constant_baselines']:raise ValueError('Constant baseline is not allowed in this selection')
        for name,digest in m['runtime_sha256'].items():
            if sha(root/name)!=digest:raise ValueError('Runtime feature/architecture fingerprint differs: '+name)
        self.run_id=m['selection_id'];self.tasks={(t['code'],t['horizon']):t for t in m['tasks']}
        if set(self.tasks)!={(c,h) for c in codes for h in m['horizons']}:raise ValueError('Daily selection incomplete')
        self.models={}
        for task in m['tasks']:
            for path in task['bundles']:
                if path in self.models:continue
                candidate=(root/path).resolve()
                if not candidate.is_relative_to((root/'code/models').resolve()):raise ValueError('Model path escapes models directory')
                if sha(candidate)!=m['bundle_sha256'][path]:raise ValueError('Selected model bundle checksum differs')
                if task['profile']=='neural':
                    from daily.neural_predictor import NeuralPredictor
                    self.models[path]=NeuralPredictor(candidate)
                elif task['profile']=='legacy':self.models[path]=RegressionPredictor(candidate,m['bundle_sha256'][path])
                else:self.models[path]=EnhancedPredictor(candidate)
        super().__init__(root,codes,m['database_sha256'],store)

    def apply(self,rows,calendar,prices):
        from daily.neural_features import pack
        frames={};raw={};memo={};missing=0;stale=0;errors={}
        index=np.asarray(calendar)
        for profile,builder in [('rich',rich_frames),('cycle',cycle_frames)]:
            if any(t['profile']==profile for t in self.tasks.values()):frames[profile]=builder(self.bars,self.codes,calendar)
        if any(t['profile']=='neural' for t in self.tasks.values()):
            raw={c:self.bars[c].set_index('date').reindex(calendar)[['open','high','low','close','volume']].to_numpy(dtype=float) for c in self.codes}
        for row in rows:
            code=row['code'];row['forecast_as_of']={};row['forecast_errors']={};row['model_ids']={};row['model_validation_mae_pp']={}
            for h in (1,5,21,63,126):
                task=self.tasks[(code,h)];profile=task['profile'];key=str(h)
                row['models'][key]=task['label'];row['model_ids'][key]=task['id'];row['model_validation_mae_pp'][key]=task['validation_mae_pp']
                bundles=[self.models[p].bundle for p in task['bundles']]
                minimum=max(max(b['metadata']['fit_label_ends'].values()) if profile=='neural' else b['metadata']['fit_label_end'] for b in bundles)
                try:
                    if profile in ('rich','cycle'):
                        f=frames[profile][code]
                        possible=f.index[(f.index>=minimum)&(f.index<=row['as_of'])&np.isfinite(f.to_numpy()).all(axis=1)].tolist()
                    elif profile=='neural':
                        length=task['lookback'];data=raw[code]
                        good=np.isfinite(data).all(axis=1)&(data>0).all(axis=1);bad=np.r_[0,np.cumsum(~good)]
                        ends=np.arange(length,len(index))
                        eligible=(bad[ends+1]-bad[ends-length])==0
                        possible=index[ends[eligible&(index[ends]>=minimum)&(index[ends]<=row['as_of'])]].tolist()
                    else:
                        known_dates=set(self.bars[code].date)
                        possible=[d for d in calendar if minimum<=d<=row['as_of'] and d in known_dates]
                    if not possible:raise ValueError('已训练模型所需的连续历史尚未就绪')
                    date=possible[-1]
                    candle=self.bars[code].set_index('date')
                    if date not in candle.index or date not in prices[code] or abs(float(candle.loc[date,'close'])-prices[code][date])>.020001:
                        raise ValueError('模型输入收盘价与行情不一致')
                    cache_key=(tuple(task['bundles']),code,date)
                    if cache_key not in memo:
                        if profile=='neural':
                            end=int(np.searchsorted(index,date));length=task['lookback']
                            batch_codes=[c for c in self.codes if end>=length and np.isfinite(raw[c][end-length:end+1]).all() and (raw[c][end-length:end+1]>0).all()]
                            x=np.concatenate([pack(raw[c],np.array([end]),length) for c in batch_codes])
                            members=[]
                            for path in task['bundles']:
                                model=self.models[path];identities=[model.bundle['codes'].index(c) for c in batch_codes]
                                members.append(model.predict(x,identities))
                            for c,prediction in zip(batch_codes,np.mean(members,axis=0)):
                                memo[(tuple(task['bundles']),c,date)]=prediction
                        elif profile in ('rich','cycle'):
                            feature=frames[profile][code].loc[date].to_dict()
                            memo[cache_key]=float(self.models[task['bundles'][0]].predict([feature])[0])
                        else:
                            dates=[d for d in calendar if d<=date][-253:]
                            if len(dates)!=253 or any(d not in prices[code] for d in dates):raise ValueError('价格历史不连续')
                            feature=one_day_features([prices[code][d] for d in dates],252)
                            memo[cache_key]=self.models[task['bundles'][0]].predict(feature)
                    value=memo[cache_key][(1,5,21,63,126).index(h)] if profile=='neural' else memo[cache_key]
                    if not np.isfinite(value):raise ValueError('模型输出无效')
                    row['forecast'][key]=float(value)*100;row['forecast_as_of'][key]=date
                    if date<row['as_of']:
                        stale+=1;row['forecast_errors'][key]='量价未同步，保留最后可用收盘预测'
                except (ValueError,KeyError,IndexError) as exc:
                    row['forecast'][key]=None;row['forecast_as_of'][key]=None;missing+=1
                    row['forecast_errors'][key]=str(exc)
            if row['forecast_errors'] or code in self.errors:
                errors[code]=dict(row['forecast_errors'])
                if code in self.errors:errors[code]['ohlcv']=self.errors[code]
        return {'selection_id':self.run_id,'policy':self.manifest['policy'],'configured_tasks':40,
                'available_tasks':40-missing,'stale_tasks':stale,'missing_tasks':missing,'errors':errors,
                'constant_baselines':0,'status':'ready' if not missing and not stale else 'partial_or_stale'}
