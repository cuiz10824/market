"""Purged rolling validation and annual forward tests for one broad index."""
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,log_loss

from bear.dataset import features,targets,ma_state,cycles,split,NAMES
from bear.models import fit,predict,save,load

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def scores(y,probability):
    prediction=probability.argmax(axis=1)
    return dict(n=len(y),accuracy=float(accuracy_score(y,prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y,prediction)),
                macro_f1=float(f1_score(y,prediction,labels=[0,1,2],average='macro',zero_division=0)),
                log_loss=float(log_loss(y,probability,labels=[0,1,2])),
                brier=float(np.mean(np.sum((probability-np.eye(3)[y])**2,axis=1))),
                class_counts=np.bincount(y,minlength=3).tolist(),
                confusion_matrix=confusion_matrix(y,prediction,labels=[0,1,2]).tolist())


def prepare():
    p=read(ROOT/'protocol.json');pointer=read(ROOT/'data/latest.json')
    manifest=read(ROOT/pointer['manifest'])
    if sha(ROOT/pointer['csv'])!=manifest['csv_sha256']:raise ValueError('Dataset checksum differs')
    frame=pd.read_csv(ROOT/pointer['csv'],dtype={'date':str})
    x=features(frame);ma=ma_state(frame)
    ys={h:targets(frame,h,p['thresholds'][str(h)],p['path_occupancy']) for h in p['horizons']}
    past={h:ys[h].label.shift(h).fillna(-1).to_numpy(dtype=int) for h in p['horizons']}
    return p,pointer,manifest,frame,x,ma,ys,past


def run():
    p,pointer,manifest,frame,x,ma,ys,past=prepare()
    source={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'train.py',ROOT/'collect.py',*sorted((ROOT/'bear').glob('*.py'))]}
    fingerprint=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    run_id=manifest['dataset_id']+'_'+sha(ROOT/'protocol.json')[:8]+'_'+fingerprint[:8]
    folder=ROOT/'reports'/run_id;model_folder=ROOT/'models'/run_id
    if folder.exists() or model_folder.exists():raise ValueError('Run exists: refuse to overwrite frozen results')
    folder.mkdir(parents=True)
    write(folder/'frozen_protocol.json',p)
    provenance=dict(run_id=run_id,dataset=pointer,dataset_sha256=manifest['csv_sha256'],
                    protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=source,
                    versions={n:importlib.metadata.version(n) for n in ('numpy','pandas','scikit-learn','lightgbm')},
                    created_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),research_only=True)
    write(folder/'provenance.json',provenance)
    entries=[];validation={};choices={};records=[];latest=[]

    def train_save(h,kind,role,cutoff,train,rows):
        if len(train)<500:raise ValueError('Insufficient training history')
        target=ys[h]
        if cutoff and not (target.label_end.iloc[train]<cutoff).all():raise ValueError('Future label leaked across training cutoff')
        model=fit(kind,x.iloc[train],target.label.iloc[train].to_numpy(),p)
        meta=dict(provenance,horizon=h,role=role,cutoff=cutoff,features=list(x.columns),
                  train_rows=len(train),train_origin_start=str(frame.date.iloc[train[0]]),
                  train_origin_end=str(frame.date.iloc[train[-1]]),train_label_end=str(target.label_end.iloc[train].max()),
                  target_threshold=p['thresholds'][str(h)],path_occupancy=p['path_occupancy'])
        bundle=save(model,model_folder/role/str(h)/kind/(cutoff or 'all'),meta)
        value=predict(model,x.iloc[rows],past[h][rows],ma[rows])
        restored=load(bundle)
        np.testing.assert_allclose(predict(restored,x.iloc[rows],past[h][rows],ma[rows]),value,atol=1e-12,rtol=1e-10)
        artifact=bundle.parent/'predictions.npz'
        np.savez_compressed(artifact,rows=rows,probabilities=value)
        entries.append(dict(horizon=h,candidate=kind,role=role,cutoff=cutoff,
                            bundle=str(bundle.relative_to(ROOT)),predictions=str(artifact.relative_to(ROOT))))
        return value

    for h in p['horizons']:
        folds=[]
        for start,end in p['validation_windows']:
            train=split(frame,x,ys[h],start,train=True);rows=split(frame,x,ys[h],start,end)
            fold=dict(start=start,end=end,train_n=len(train),validation_n=len(rows),
                      train_last_label=ys[h].label_end.iloc[train].max(),scores={})
            for kind in p['candidates']:
                value=train_save(h,kind,'validation',start,train,rows)
                fold['scores'][kind]=scores(ys[h].label.iloc[rows].to_numpy(),value)
            folds.append(fold)
            print(f'h={h} validation {start} complete',flush=True)
        validation[str(h)]=folds
        means={kind:float(np.mean([f['scores'][kind]['macro_f1'] for f in folds])) for kind in p['candidates']}
        choices[str(h)]=dict(candidate=max(p['candidates'],key=lambda k:means[k]),validation_macro_f1=means)
    # Freeze all selections before fitting any final test or latest model.
    write(folder/'choices.json',dict(choices=choices,validation=validation))
    print('CHOICES FROZEN '+json.dumps(choices),flush=True)
    test_scores={}
    for h in p['horizons']:
        target=ys[h]
        for year in p['test_years']:
            start=f'{year}-01-01';end=f'{year+1}-01-01'
            train=split(frame,x,target,start,train=True)
            # Annual forecast origins; their realized labels may end in the following year.
            rows=np.flatnonzero((x.notna().all(axis=1)&(frame.date>=start)&(frame.date<end)&(target.label>=0)).to_numpy())
            if not len(rows):continue
            for kind in p['candidates']:
                value=train_save(h,kind,'test',start,train,rows)
                for k,ix in enumerate(rows):
                    records.append(dict(date=frame.date.iloc[ix],horizon=h,label_end_date=target.label_end.iloc[ix],
                         candidate=kind,actual_class=int(target.label.iloc[ix]),predicted_class=int(value[k].argmax()),
                         bear_score=float(value[k,0]),mixed_score=float(value[k,1]),bull_score=float(value[k,2]),
                         actual_return=float(target.forward_return.iloc[ix]),selected=kind==choices[str(h)]['candidate']))
            print(f'h={h} test {year} complete',flush=True)
        test_scores[str(h)]={}
        for kind in p['candidates']:
            group=[r for r in records if r['horizon']==h and r['candidate']==kind]
            test_scores[str(h)][kind]=scores(np.asarray([r['actual_class'] for r in group]),
                 np.asarray([[r['bear_score'],r['mixed_score'],r['bull_score']] for r in group]))
            train=split(frame,x,target,None,train=True);rows=np.array([len(frame)-1])
            value=train_save(h,kind,'latest',None,train,rows)[0]
            latest.append(dict(horizon=h,candidate=kind,as_of=frame.date.iloc[-1],
                               predicted_class=int(value.argmax()),label=NAMES[int(value.argmax())],
                               scores=value.tolist(),selected=kind==choices[str(h)]['candidate']))
    csvwrite(folder/'test_predictions.csv',records)
    frame.assign(past_only_state=ma).to_csv(folder/'past_only_trend.csv',index=False,lineterminator='\n')
    cycles(frame,p['retrospective_reversal']).to_csv(folder/'historical_cycles.csv',index=False,lineterminator='\n')
    annual=[]
    for h in p['horizons']:
        for kind in p['candidates']:
            for year in p['test_years']:
                group=[r for r in records if r['horizon']==h and r['candidate']==kind and r['date'].startswith(str(year))]
                if group:
                    m=scores(np.asarray([r['actual_class'] for r in group]),np.asarray([[r['bear_score'],r['mixed_score'],r['bull_score']] for r in group]))
                    annual.append(dict(horizon=h,candidate=kind,origin_year=year,**{k:v for k,v in m.items() if not isinstance(v,list)}))
    csvwrite(folder/'yearly_metrics.csv',annual)
    summary=dict(provenance,protocol=p,choices=choices,validation=validation,test_scores=test_scores,
                 latest=latest,last_date=frame.date.iloc[-1],rows=len(frame),feature_count=len(x.columns),
                 first_feature_date=frame.date[x.notna().all(axis=1)].iloc[0],class_names=NAMES)
    write(folder/'summary.json',summary)
    write(model_folder/'index.json',dict(run_id=run_id,models=entries))
    write(folder/'checksums.json',{str(f.relative_to(ROOT)):sha(f) for f in model_folder.rglob('*') if f.is_file()})
    write(ROOT/'models/latest.json',dict(run_id=run_id,index=str((model_folder/'index.json').relative_to(ROOT)),
               report=str((folder/'summary.json').relative_to(ROOT))))
    print('RUN_ID='+run_id,flush=True)


if __name__=='__main__':run()
