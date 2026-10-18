"""Shared small-network CPU experiment; all choices frozen before held-out scoring."""
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html')]
from research.sequence_data import SequenceData,CHANNELS
from training.neural_engine import fit,predict,scaled_score,export,load
from training.train_enhanced import read,write,sha,metrics,old_reports


def group_scores(data,rows,mask,values):
    result={}
    for ci,code in enumerate(data.codes):
        for j,h in enumerate(data.horizons):
            valid=(data.ids[rows]==ci)&mask[:,j]
            if valid.sum()==0:raise ValueError('Empty index/horizon score')
            result[f'{code}:{h}']=metrics(data.y[rows[valid],j],values[valid,j])
    return result


def legacy_validation(old,code,h,start):
    task=old[(code,h)];selected=task.get('choice') if h==1 else task['choices']['return']
    fold=next(f for f in task['development_folds'] if f['start']==start and f['status']=='completed')
    return fold['scores'][selected]['mae']*100 if h==1 else fold['scores']['return'][selected]*100


def run():
    path=ROOT/'code/training/neural_protocol.json';p=read(path)
    torch.set_num_threads(p['threads']);torch.set_num_interop_threads(1)
    data=SequenceData(p);old,original_predictions,_=old_reports()
    # All existing model/report bytes are immutable; a new experiment gets a new namespace.
    locked={str(f.relative_to(ROOT)):sha(f) for folder in ('code/models','data/reports/training') for f in (ROOT/folder).rglob('*') if f.is_file()}
    for name in ('data/market.sqlite3','data/enhanced/market_ohlcv.sqlite3'):locked[name]=sha(ROOT/name)
    sources=['code/training/train_neural.py','code/training/neural_engine.py','code/research/sequence_data.py',
             'code/research/neural_models.py','code/training/train_enhanced.py','data/active_indices.json']
    source_hashes={s:sha(ROOT/s) for s in sources}
    fingerprint=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    run_id='neural_'+dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d')+'_'+sha(path)[:8]+'_'+fingerprint[:8]
    models=ROOT/'code/models'/run_id;report=ROOT/'data/reports/training'/run_id
    if models.exists() or report.exists():raise ValueError('Run already exists; do not overwrite')
    provenance={'run_id':run_id,'protocol_sha256':sha(path),'source_sha256':source_hashes,
                'database_sha256':sha(ROOT/'data/enhanced/market_ohlcv.sqlite3'),'research_only':True,
                'versions':{n:importlib.metadata.version(n) for n in ('torch','numpy','pandas','safetensors')}}
    write(report/'frozen_protocol.json',p);write(report/'provenance.json',provenance)
    entries=[];screening={};chosen={};epochs={};development={};records=[];latest={}
    cache={}

    def train_save(architecture,length,seed,role,start,end=None,fixed_epochs=None):
        train,tm=data.split(start,train=True,stride=p['train_stride'])
        validation=None
        if role in ('screen','replication'):
            validation=data.split(start,end,stride=p['early_stop_validation_stride'])
        if len(train)<500:raise ValueError('Insufficient pooled training origins')
        net,log=fit(data,architecture,length,seed,train,tm,p,validation=validation,epochs=fixed_epochs)
        tag=f'{role}/{architecture}/{length}/{start or "all"}/{seed}'
        folder=models/tag
        metadata=dict(provenance,role=role,seed=seed,cutoff=start,fit_origin_start=str(data.dates[train].min()),
                      fit_origin_end=str(data.dates[train].max()),training_rows=len(train),
                      label_counts={str(h):int(tm[:,j].sum()) for j,h in enumerate(data.horizons)},
                      fit_label_ends={str(h):str(data.calendar[data.label_indices[train,j][tm[:,j]]].max()) for j,h in enumerate(data.horizons)},
                      selected_epoch=log['selected_epoch'],epochs_executed=log['epochs_executed'],parameter_count=log['parameter_count'])
        spec=dict(architecture=architecture,lookback=length,sequence_length=data.x[length].shape[1],channels=list(CHANNELS),
                  codes=data.codes,horizons=data.horizons,target_scales=data.target_scales.tolist(),metadata=metadata)
        bundle=export(net,folder,spec)
        log_path=report/'fit_logs'/(tag.replace('/','_')+'.json');write(log_path,log)
        entry=dict(role=role,architecture=architecture,lookback=length,seed=seed,start=start,end=end,
                   bundle=str(bundle.relative_to(ROOT)),log=str(log_path.relative_to(ROOT)))
        if validation is not None:
            rows,mask=data.split(start,end)
            values=predict(net,data,length,rows);reloaded,_=load(bundle)
            np.testing.assert_allclose(predict(reloaded,data,length,rows),values,atol=1e-10,rtol=1e-8)
            artifact=folder/'validation.npz';np.savez_compressed(artifact,rows=rows,mask=mask,prediction=values)
            entry['validation']=str(artifact.relative_to(ROOT));entry['score']=scaled_score(data,rows,mask,values)
            entry['selected_epoch']=log['selected_epoch'];entry['task_scores']=group_scores(data,rows,mask,values)
        entries.append(entry)
        print(f'{tag} complete: epochs={log["epochs_executed"]}, selected={log["selected_epoch"]}',flush=True)
        return net,entry

    # First seed screens lengths; no held-out predictions are made during screening.
    for architecture in p['architectures']:
        screening[architecture]={}
        for length in p['lookbacks']:
            folds=[]
            for start,end in p['development_windows']:
                _,entry=train_save(architecture,length,p['screening_seed'],'screen',start,end)
                folds.append(entry)
            screening[architecture][str(length)]=float(np.mean([e['score'] for e in folds]))
        length=min(p['lookbacks'],key=lambda n:screening[architecture][str(n)]);chosen[architecture]=length
        print(f'LOOKBACK {architecture}={length}, scores={screening[architecture]}',flush=True)
    write(report/'lookback_choices.json',dict(screening=screening,chosen=chosen))
    for architecture in p['architectures']:
        length=chosen[architecture];folds=[];epoch_counts=[]
        for start,end in p['development_windows']:
            _,second=train_save(architecture,length,p['replication_seed'],'replication',start,end)
            first=next(e for e in entries if e['role']=='screen' and e['architecture']==architecture and e['lookback']==length and e['start']==start)
            with np.load(ROOT/first['validation'],allow_pickle=False) as a:rows=a['rows'];mask=a['mask'];one=a['prediction']
            with np.load(ROOT/second['validation'],allow_pickle=False) as a:
                assert np.array_equal(rows,a['rows']) and np.array_equal(mask,a['mask']);two=a['prediction']
            train,tm=data.split(start,train=True,stride=p['train_stride'])
            baseline=np.zeros((len(rows),len(data.horizons)),dtype=float)
            for ci in range(len(data.codes)):
                for j in range(len(data.horizons)):
                    eligible=(data.ids[train]==ci)&tm[:,j]
                    baseline[data.ids[rows]==ci,j]=float(np.mean(data.y[train[eligible],j]))
            folds.append(dict(start=start,end=end,ensemble=group_scores(data,rows,mask,(one+two)/2),
                              zero=group_scores(data,rows,mask,np.zeros_like(one)),mean=group_scores(data,rows,mask,baseline)))
            epoch_counts.extend([first['selected_epoch'],second['selected_epoch']])
        epochs[architecture]=int(np.floor(np.median(epoch_counts)+.5));development[architecture]=folds
    # Freeze the ensemble decisions and refit budget before any held-out prediction.
    tasks=[]
    for code in data.codes:
        for h in data.horizons:
            key=f'{code}:{h}';means={a:float(np.mean([f['ensemble'][key]['mae_pp'] for f in development[a]])) for a in p['architectures']}
            reference=development[p['architectures'][0]]
            means['zero']=float(np.mean([f['zero'][key]['mae_pp'] for f in reference]))
            means['mean']=float(np.mean([f['mean'][key]['mae_pp'] for f in reference]))
            means['legacy']=float(np.mean([legacy_validation(old,code,h,f['start']) for f in reference]))
            winner=min(p['architectures'],key=lambda a:means[a]);gain=1-means[winner]/min(means[k] for k in ('zero','mean','legacy'))
            choice=winner if gain>=p['min_relative_validation_gain'] else 'legacy'
            tasks.append(dict(code=code,horizon=h,choice=choice,validation_mean_mae_pp=means,relative_validation_gain=gain))
    write(report/'choices.json',dict(tasks=tasks,chosen_lookbacks=chosen,refit_epochs=epochs,development=development))
    for architecture in p['architectures']:
        length=chosen[architecture]
        for start in p['holdout_starts']:
            rows,mask=data.split(start)
            targets=['000510'] if start==p['a500_holdout_start'] else [c for c in data.codes if c!='000510']
            subset=np.isin(data.ids[rows],[data.codes.index(c) for c in targets]);rows=rows[subset];mask=mask[subset]
            members=[]
            for seed in (p['screening_seed'],p['replication_seed']):
                net,entry=train_save(architecture,length,seed,'holdout',start,fixed_epochs=epochs[architecture])
                values=predict(net,data,length,rows);reloaded,_=load(ROOT/entry['bundle'])
                np.testing.assert_allclose(predict(reloaded,data,length,rows),values,atol=1e-10,rtol=1e-8)
                artifact=(ROOT/entry['bundle']).parent/'test.npz';np.savez_compressed(artifact,rows=rows,mask=mask,prediction=values)
                entry['test']=str(artifact.relative_to(ROOT));members.append(values)
            ensemble=(members[0]+members[1])/2
            for k,ix in enumerate(rows):
                code=data.codes[data.ids[ix]];date=str(data.dates[ix])
                for j,h in enumerate(data.horizons):
                    if not mask[k,j]:continue
                    reference=original_predictions[(code,h,date)]
                    actual=float(data.y[ix,j])
                    if abs(actual-float(reference['actual_return']))>1e-12:raise ValueError('Neural target differs from old observed price target')
                    records.append(dict(code=code,horizon=h,date=date,label_end_date=str(data.calendar[data.label_indices[ix,j]]),
                                        candidate=architecture,lookback=length,actual_return=actual,expected_return=float(ensemble[k,j]),
                                        seed1_return=float(members[0][k,j]),seed2_return=float(members[1][k,j]),
                                        legacy_return=float(reference['expected_return']),zero_return=0.))
        live=np.array([np.where((data.ids==ci)&(data.ends==len(data.calendar)-1))[0][0] for ci in range(len(data.codes))])
        members=[]
        for seed in (p['screening_seed'],p['replication_seed']):
            net,entry=train_save(architecture,length,seed,'latest',None,fixed_epochs=epochs[architecture])
            value=predict(net,data,length,live);restored,_=load(ROOT/entry['bundle'])
            np.testing.assert_allclose(predict(restored,data,length,live),value,atol=1e-10,rtol=1e-8)
            artifact=(ROOT/entry['bundle']).parent/'latest.npz';np.savez_compressed(artifact,rows=live,prediction=value)
            entry['latest']=str(artifact.relative_to(ROOT));members.append(value)
        latest[architecture]=(members[0]+members[1])/2
    for t in tasks:
        code=t['code'];h=t['horizon'];ci=data.codes.index(code);j=data.horizons.index(h)
        group=[r for r in records if r['code']==code and r['horizon']==h]
        t['test_candidates']={a:metrics([r['actual_return'] for r in group if r['candidate']==a],[r['expected_return'] for r in group if r['candidate']==a]) for a in p['architectures']}
        reference=[r for r in group if r['candidate']==p['architectures'][0]]
        t['test_legacy']=metrics([r['actual_return'] for r in reference],[r['legacy_return'] for r in reference])
        t['test_zero']=metrics([r['actual_return'] for r in reference],[0.]*len(reference))
        t['latest_candidate_returns']={a:float(latest[a][ci,j]) for a in p['architectures']}
    for name,digest in locked.items():
        if sha(ROOT/name)!=digest:raise ValueError('Original artifact changed: '+name)
    summary=dict(provenance,protocol=p,tasks=tasks,screening=screening,chosen_lookbacks=chosen,refit_epochs=epochs,
                 development=development,origin_count=len(data.ids),latest_as_of=str(data.calendar[-1]))
    write(report/'summary.json',summary);write(models/'index.json',dict(provenance,models=entries,tasks=tasks,auto_deploy=False))
    with (report/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    saved={str(f.relative_to(ROOT)):sha(f) for f in models.rglob('*') if f.is_file()}
    write(report/'verification.json',dict(status='training_export_parity_passed',native_bundles=len(entries),candidate_predictions=len(records),
                                        all_saved_file_sha256=saved,original_artifact_sha256=locked,original_artifacts_unchanged=True))
    write(ROOT/'code/models/latest_neural.json',dict(run_id=run_id,research_only=True,auto_deploy=False,index=str((models/'index.json').relative_to(ROOT)),
          report=str((report/'summary.json').relative_to(ROOT)),verification=str((report/'verification.json').relative_to(ROOT))))
    print('RUN_ID='+run_id,flush=True)


if __name__=='__main__':run()
