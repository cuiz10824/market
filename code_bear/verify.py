"""Reload all native models, reproduce split predictions and recompute test metrics."""
import json
import datetime as dt
import gzip
import hashlib
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from train import ROOT,read,sha,write,scores
from bear.dataset import split,features,targets,ma_state
from bear.models import load,predict
from collect import parse


def verify():
    pointer=read(ROOT/'models/latest.json');s=read(ROOT/pointer['report'])
    folder=(ROOT/pointer['report']).parent
    for name,digest in s['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Research source changed:'+name)
    for name,digest in read(folder/'checksums.json').items():
        if sha(ROOT/name)!=digest:raise ValueError('Saved model changed:'+name)
    p=s['protocol'];manifest=read(ROOT/s['dataset']['manifest'])
    frame=pd.read_csv(ROOT/s['dataset']['csv'])
    if sha(ROOT/s['dataset']['csv'])!=s['dataset_sha256']:raise ValueError('Frozen CSV changed')
    rebuilt={}
    snapshot=(ROOT/s['dataset']['manifest']).parent
    for source in manifest['sources']:
        raw=gzip.decompress((snapshot/'raw'/f'{source["year"]}.json.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest()!=source['response_sha256']:raise ValueError('Raw response changed')
        for row in parse(json.loads(raw),dt.datetime.fromisoformat(manifest['retrieved_at'])):
            if row['date'].startswith(str(source['year'])):rebuilt[row['date']]=row
    original=pd.DataFrame([rebuilt[d] for d in sorted(rebuilt)])
    np.testing.assert_array_equal(frame.date,original.date)
    np.testing.assert_allclose(frame.iloc[:,1:],original.iloc[:,1:],rtol=1e-12,atol=1e-6)
    with sqlite3.connect((snapshot/'market.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as database:
        saved=pd.read_sql_query('SELECT * FROM bars ORDER BY date',database)
    np.testing.assert_array_equal(saved.date,frame.date)
    np.testing.assert_allclose(saved.iloc[:,1:],frame.iloc[:,1:],rtol=1e-12,atol=1e-6)
    x=features(frame);ma=ma_state(frame)
    ys={h:targets(frame,h,p['thresholds'][str(h)],p['path_occupancy']) for h in p['horizons']}
    past={h:ys[h].label.shift(h).fillna(-1).to_numpy(dtype=int) for h in p['horizons']}
    if manifest['csv_sha256']!=s['dataset_sha256']:raise ValueError('Dataset differs')
    index=read(ROOT/pointer['index']);csv=pd.read_csv(folder/'test_predictions.csv')
    count=0;test_count=0
    for entry in index['models']:
        model=load(ROOT/entry['bundle']);h=entry['horizon'];kind=entry['candidate'];role=entry['role'];start=entry['cutoff']
        train=split(frame,x,ys[h],start,train=True)
        m=model['metadata']
        assert len(train)==m['train_rows']
        assert str(ys[h].label_end.iloc[train].max())==m['train_label_end']
        if start:assert (ys[h].label_end.iloc[train]<start).all()
        if role=='validation':
            _,end=next(w for w in p['validation_windows'] if w[0]==start)
            rows=split(frame,x,ys[h],start,end)
        elif role=='test':
            end=f'{int(start[:4])+1}-01-01'
            rows=np.flatnonzero((x.notna().all(axis=1)&(frame.date>=start)&(frame.date<end)&(ys[h].label>=0)).to_numpy())
        else:rows=np.array([len(frame)-1])
        with np.load(ROOT/entry['predictions'],allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['rows'],rows)
            value=predict(model,x.iloc[rows],past[h][rows],ma[rows])
            np.testing.assert_allclose(value,saved['probabilities'],atol=1e-12,rtol=1e-10)
        count+=1
        if role=='test':
            part=csv[(csv.horizon==h)&(csv.candidate==kind)&(csv.date>=start)&(csv.date<end)]
            np.testing.assert_array_equal(part.date,frame.date.iloc[rows])
            np.testing.assert_array_equal(part.actual_class,ys[h].label.iloc[rows])
            np.testing.assert_array_equal(part.label_end_date,ys[h].label_end.iloc[rows])
            np.testing.assert_allclose(part[['bear_score','mixed_score','bull_score']],value,atol=1e-12,rtol=1e-10)
            origin=frame.close.iloc[rows].to_numpy();future=frame.close.iloc[rows+h].to_numpy()
            np.testing.assert_allclose(part.actual_return,future/origin-1,atol=1e-12,rtol=1e-10)
            truth=[]
            for ix in rows:
                base=frame.close.iloc[ix];path=frame.close.iloc[ix+1:ix+h+1].to_numpy()
                change=path[-1]/base-1;threshold=p['thresholds'][str(h)]
                truth.append(2 if change>=threshold and np.mean(path>base)>=p['path_occupancy'] else
                             0 if change<=-threshold and np.mean(path<base)>=p['path_occupancy'] else 1)
            np.testing.assert_array_equal(part.actual_class,truth)
            test_count+=len(part)
    assert test_count==len(csv)
    for h in p['horizons']:
        means={k:np.mean([f['scores'][k]['macro_f1'] for f in s['validation'][str(h)]]) for k in p['candidates']}
        assert s['choices'][str(h)]['candidate']==max(p['candidates'],key=lambda k:means[k])
        for kind in p['candidates']:
            part=csv[(csv.horizon==h)&(csv.candidate==kind)]
            actual=scores(part.actual_class.to_numpy(),part[['bear_score','mixed_score','bull_score']].to_numpy())
            for name in ('accuracy','balanced_accuracy','macro_f1','log_loss','brier'):
                np.testing.assert_allclose(actual[name],s['test_scores'][str(h)][kind][name],atol=1e-12)
    result=dict(status='passed',native_models=count,test_csv_rows=test_count,
                raw_responses_verified=len(manifest['sources']),sqlite_csv_match=True,
                labels_recomputed=True,purged_cutoffs=True,selection_validation_only=True,
                dataset_sha256=s['dataset_sha256'])
    write(folder/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':verify()
