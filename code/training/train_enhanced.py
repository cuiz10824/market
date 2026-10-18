"""OHLCV + cross-index CPU experiment; freeze validation choices before test scoring."""
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html')]
from daily.enhanced_predictor import EnhancedPredictor
from daily.rich_features import feature_names
from research.rich_dataset import load_bars,samples
from training.enhanced_models import fit,predict,export


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def metrics(actual,prediction):
    e=(np.asarray(prediction)-np.asarray(actual))*100
    return {'n':len(e),'mae_pp':float(np.mean(np.abs(e))),'bias_pp':float(np.mean(e)),
            'error_std_pp':float(np.std(e,ddof=0)),'rmse_pp':float(np.sqrt(np.mean(e**2)))}


def old_reports():
    tasks={};predictions={};locked={}
    for pointer_name in ('latest.json','latest_one_day.json','latest_three_class.json'):
        path=ROOT/'code/models'/pointer_name
        pointer=read(path);locked[str(path.relative_to(ROOT))]=sha(path)
        model_folder=(ROOT/pointer['index']).parent
        report=ROOT/'data/reports/training'/pointer['run_id']
        for f in list(model_folder.rglob('*'))+list(report.rglob('*')):
            if f.is_file():locked[str(f.relative_to(ROOT))]=sha(f)
        if pointer_name=='latest_three_class.json':continue
        for t in read(report/'summary.json')['tasks']:tasks[(t['code'],t['horizon'])]=t
        with (report/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):predictions[(r['code'],int(r['horizon']),r['date'])]=r
    return tasks,predictions,locked


def run():
    protocol_path=ROOT/'code/training/enhanced_protocol.json';p=read(protocol_path)
    codes=p['codes'];names=feature_names(codes)
    bars,calendar=load_bars();datasets,feature_frames=samples(bars,codes,calendar)
    if len(names)!=p['feature_count']:raise ValueError('Feature count differs from frozen protocol')
    old,old_predictions,locked=old_reports()
    sources=['code/training/train_enhanced.py','code/training/enhanced_models.py','code/research/rich_dataset.py',
             'html/daily/rich_features.py','html/daily/enhanced_predictor.py','data/active_indices.json']
    source_hashes={s:sha(ROOT/s) for s in sources}
    fingerprint=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    data_sha=sha(ROOT/'data/enhanced/market_ohlcv.sqlite3')
    date=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d')
    run_id=f'enhanced_{date}_{sha(protocol_path)[:8]}_{data_sha[:8]}_{fingerprint[:8]}'
    models=ROOT/'code/models'/run_id;report=ROOT/'data/reports/training'/run_id
    if models.exists() or report.exists():raise ValueError('Run exists, do not overwrite')
    provenance={'run_id':run_id,'database_sha256':data_sha,'source_sha256':source_hashes,
                'protocol_sha256':sha(protocol_path),'test_status':p['test_status'],
                'versions':{name:importlib.metadata.version(name) for name in ('numpy','pandas','lightgbm','catboost')},
                'research_only':True}
    write(report/'frozen_protocol.json',p);write(report/'provenance.json',provenance)
    tasks=[];records=[];entries=[];candidate_latest={};all_folds={}
    for h,frame in datasets.items():
        folds=[]
        for start,end in p['development_windows']:
            train=frame[frame.label_end_date<start]
            valid=frame[(frame.date>=start)&(frame.date<end)&(frame.label_end_date<end)]
            if len(train)<p['min_train_rows']:raise ValueError('Insufficient pooled training history')
            result={c:{} for c in codes}
            for kind in p['candidates']:
                model=fit(kind,train,names,p);values=predict(model,kind,valid,names)
                for code in codes:
                    mask=(valid.code==code).to_numpy()
                    if mask.sum()<p['min_validation_rows_per_index']:raise ValueError('Insufficient validation observations')
                    result[code][kind]=metrics(valid.loc[mask,'target_return'],values[mask])
            for code in codes:
                actual=valid.loc[valid.code==code,'target_return']
                mean=float(train.loc[train.code==code,'target_return'].mean())
                result[code]['zero']=metrics(actual,np.zeros(len(actual)))
                result[code]['mean']=metrics(actual,np.full(len(actual),mean))
                previous=old[(code,h)]
                selected=previous.get('choice') if h==1 else previous['choices']['return']
                previous_fold=next(f for f in previous['development_folds'] if f['start']==start and f['status']=='completed')
                result[code]['legacy_mae_pp']=(previous_fold['scores'][selected]['mae']*100 if h==1
                                              else previous_fold['scores']['return'][selected]*100)
            folds.append({'start':start,'end':end,'train_n':len(train),'last_train_label_end':train.label_end_date.max(),
                          'validation_n':len(valid),'results':result})
            print(f'h={h} development {start} done',flush=True)
        all_folds[str(h)]=folds
        choices={};means={}
        for code in codes:
            means[code]={kind:float(np.mean([f['results'][code][kind]['mae_pp'] for f in folds]))
                         for kind in p['candidates']+['zero','mean']}
            means[code]['legacy']=float(np.mean([f['results'][code]['legacy_mae_pp'] for f in folds]))
            winner=min(p['candidates'],key=lambda k:means[code][k])
            baseline=min(means[code][k] for k in ('zero','mean','legacy'))
            gain=1-means[code][winner]/baseline
            choices[code]=winner if gain>=p['min_relative_validation_gain'] else 'legacy'
        # Freeze all 8 choices before any test prediction or metric is inspected.
        write(report/f'choices_{h}.json',{'choices':choices,'validation_mean_mae_pp':means})
        test_results={code:{} for code in codes};test_rows={}
        for start in p['holdout_starts']:
            train=frame[frame.label_end_date<start]
            targets=['000510'] if start==p['a500_holdout_start'] else [c for c in codes if c!='000510']
            test=frame[(frame.date>=start)&frame.code.isin(targets)].copy()
            test_rows.update({c:test[test.code==c] for c in targets})
            for kind in p['candidates']:
                model=fit(kind,train,names,p)
                values=predict(model,kind,test,names)
                folder=models/('holdout_'+start[:4])/str(h)/kind
                folder.mkdir(parents=True)
                bundle={'schema_version':1,'task':'enriched_return','horizon':h,'features':list(names),
                        'return':export(model,kind,folder),'metadata':dict(provenance,role='holdout',
                          fit_label_end=train.label_end_date.max(),fit_feature_end=train.date.max(),holdout_start=start)}
                path=folder/'model.json';write(path,bundle)
                loaded=EnhancedPredictor(path)
                np.testing.assert_allclose(loaded.predict(test.to_dict('records')),values,atol=1e-12,rtol=1e-10)
                for code in targets:
                    mask=(test.code==code).to_numpy();rows=test.loc[mask]
                    test_results[code][kind]=metrics(rows.target_return,values[mask])
                    for r,value in zip(rows.to_dict('records'),values[mask]):
                        original=old_predictions[(code,h,r['date'])]
                        if abs(r['target_return']-float(original['actual_return']))>1e-12:raise ValueError('Test target differs from previous close history')
                        records.append({'code':code,'horizon':h,'date':r['date'],'label_end_date':r['label_end_date'],
                                        'candidate':kind,'actual_return':r['target_return'],'expected_return':float(value),
                                        'zero_return':0.,'legacy_return':float(original['expected_return'])})
                entries.append({'role':'holdout','start':start,'horizon':h,'candidate':kind,'bundle':str(path.relative_to(ROOT))})
        latest_paths={}
        for kind in p['candidates']:
            model=fit(kind,frame,names,p)
            folder=models/'latest'/str(h)/kind;folder.mkdir(parents=True)
            bundle={'schema_version':1,'task':'enriched_return','horizon':h,'features':list(names),
                    'return':export(model,kind,folder),'metadata':dict(provenance,role='latest',
                      fit_label_end=frame.label_end_date.max(),fit_feature_end=frame.date.max(),inference_as_of=calendar[-1])}
            path=folder/'model.json';write(path,bundle);latest_paths[kind]=str(path.relative_to(ROOT))
            live=[feature_frames[c].loc[calendar[-1]].to_dict() for c in codes]
            live_frame=__import__('pandas').DataFrame(live)
            loaded=EnhancedPredictor(path);values=loaded.predict(live)
            np.testing.assert_allclose(values,predict(model,kind,live_frame,names),atol=1e-12,rtol=1e-10)
            candidate_latest[(h,kind)]=dict(zip(codes,values.tolist()))
            entries.append({'role':'latest','horizon':h,'candidate':kind,'bundle':str(path.relative_to(ROOT))})
        for code in codes:
            rows=test_rows[code]
            actual=rows.target_return.to_numpy()
            legacy=np.array([float(old_predictions[(code,h,d)]['expected_return']) for d in rows.date])
            zero=metrics(actual,np.zeros(len(actual)));previous=metrics(actual,legacy)
            selected=choices[code]
            tasks.append({'code':code,'horizon':h,'choice':selected,'validation_mean_mae_pp':means[code],
                          'test_start':rows.date.min(),'test_origin_end':rows.date.max(),
                          'test_candidates':test_results[code],'test_legacy':previous,'test_zero':zero,
                          'test_selected':previous if selected=='legacy' else test_results[code][selected],
                          'latest_bundle':None if selected=='legacy' else latest_paths[selected],
                          'latest_candidate_returns':{kind:candidate_latest[(h,kind)][code] for kind in p['candidates']}})
        print(f'h={h} choices={choices}',flush=True)
    for path,digest in locked.items():
        if sha(ROOT/path)!=digest:raise ValueError('Original artifact changed: '+path)
    if sha(ROOT/'data/enhanced/market_ohlcv.sqlite3')!=data_sha:raise ValueError('Enriched database changed')
    write(models/'index.json',dict(provenance,models=entries,tasks=tasks))
    write(report/'summary.json',dict(provenance,protocol=p,tasks=tasks,development_folds=all_folds))
    with (report/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    saved={str(f.relative_to(ROOT)):sha(f) for f in models.rglob('*') if f.is_file()}
    write(report/'verification.json',{'status':'passed','saved_model_bundles':len(entries),
          'candidate_predictions_reloaded':len(records),'all_saved_file_sha256':saved,
          'original_artifact_sha256':locked,'original_artifacts_unchanged':True})
    write(ROOT/'code/models/latest_enhanced.json',{'run_id':run_id,'research_only':True,
         'index':str((models/'index.json').relative_to(ROOT)),
         'report':str((report/'summary.json').relative_to(ROOT)),'verification':str((report/'verification.json').relative_to(ROOT))})
    print('RUN_ID='+run_id,flush=True)


if __name__=='__main__':run()
