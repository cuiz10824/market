"""Train and test next-session return regressors without touching older horizons."""
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'html'))
from daily.regression import RegressionPredictor
from research.dataset import load_history
from research.one_day import FEATURES, build_one_day_samples, one_day_features
from training.engine import choose, export_spec, return_metrics, split_window
from training.one_day_models import fit, predict
from daily.selection import scoped_protocol


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run():
    protocol_path=ROOT/'code/training/one_day_protocol.json'
    p=scoped_protocol(ROOT,read(protocol_path))
    protocol_sha=hashlib.sha256((json.dumps(p,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    sources=['code/training/train_one_day.py','code/training/one_day_models.py',
             'code/research/one_day.py','code/research/dataset.py','code/training/engine.py',
             'html/daily/features.py','html/daily/regression.py','html/daily/selection.py','data/active_indices.json']
    source_sha={name:sha(ROOT/name) for name in sources}
    db_sha=sha(ROOT/'data/market.sqlite3')
    source_id=hashlib.sha256(json.dumps(source_sha,sort_keys=True).encode()).hexdigest()
    date=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d')
    run_id=f'oneday_{date}_{protocol_sha[:8]}_{db_sha[:8]}_{source_id[:8]}'
    model_root=ROOT/'code/models'/run_id
    report=ROOT/'data/reports/training'/run_id
    if model_root.exists() or report.exists():raise ValueError('Run exists; preserve existing results')
    locked={}
    for pointer_name in ('latest.json','latest_three_class.json'):
        pointer_path=ROOT/'code/models'/pointer_name
        locked[str(pointer_path.relative_to(ROOT))]=sha(pointer_path)
        pointer=read(pointer_path)
        old_root=(ROOT/pointer['index']).parent
        old_report=ROOT/'data/reports/training'/pointer['run_id']
        for path in list(old_root.rglob('*'))+list(old_report.rglob('*')):
            if path.is_file():locked[str(path.relative_to(ROOT))]=sha(path)
    provenance={'run_id':run_id,'database_sha256':db_sha,'protocol_sha256':protocol_sha,
                'source_sha256':source_sha,'python':platform.python_version(),
                'versions':{name:importlib.metadata.version(name) for name in ('numpy','scikit-learn','lightgbm')},
                'test_status':p['test_status'],'research_only':True}
    tasks=[];records=[];entries=[];yearly=[]
    for code in p['codes']:
        calendar,prices,meta=load_history(code)
        rows=build_one_day_samples(calendar,prices,code)
        scores={kind:[] for kind in p['regression_candidates']};folds=[]
        for start,end in p['development_windows']:
            split=split_window(rows,start,end,p['min_train_samples'],p['min_validation_samples'])
            if split is None:
                folds.append({'start':start,'end':end,'status':'skipped_insufficient_history'});continue
            train,valid=split
            result={kind:return_metrics(valid,predict(fit(kind,train,p),valid)) for kind in scores}
            for kind in scores:scores[kind].append(result[kind]['mae'])
            folds.append({'start':start,'end':end,'status':'completed','train_n':len(train),
                          'validation_n':len(valid),'last_train_label_end':train[-1]['label_end_date'],
                          'last_validation_label_end':valid[-1]['label_end_date'],'scores':result})
        if not scores['zero']:raise ValueError('No development folds')
        winner,averages=choose(scores,'return',p)
        start=p['holdout_overrides'].get(code,p['holdout_start'])
        assert all(end<=start for _,end in p['development_windows'])
        train,test=split_window(rows,start,min_train=p['min_train_samples'],min_test=p['min_validation_samples'])
        fitted=fit(winner,train,p)
        estimated=predict(fitted,test)
        mean_values=predict(fit('mean',train,p),test)
        test_metrics={'selected':return_metrics(test,estimated),'zero':return_metrics(test,np.zeros(len(test))),
                      'historical_mean':return_metrics(test,mean_values)}
        tail=[prices[d] for d in calendar[-253:]]
        live=dict(date=calendar[-1],**one_day_features(tail,252))
        bundles={}
        for role,model,fit_rows,check_rows in [('holdout',fitted,train,test),
                                             ('latest',fit(winner,rows,p),rows,[live])]:
            folder=model_root/role/code/'1';folder.mkdir(parents=True)
            metadata=dict(provenance,role=role,fit_samples=len(fit_rows),
                          fit_feature_end=fit_rows[-1]['date'],fit_label_end=fit_rows[-1]['label_end_date'],
                          inference_as_of=calendar[-1],index_name=meta['name'])
            bundle={'schema_version':1,'task':'next_session_return','code':code,'horizon':1,
                    'features':list(FEATURES),'return':export_spec(model,folder,'return'),'metadata':metadata}
            path=folder/'model.json';write(path,bundle)
            reloaded=RegressionPredictor(path)
            np.testing.assert_allclose([reloaded.predict(r) for r in check_rows],predict(model,check_rows),rtol=1e-10,atol=1e-12)
            bundles[role]=str(path.relative_to(ROOT))
        tasks.append({'code':code,'name':meta['name'],'horizon':1,'choice':winner,
                      'development_mean_mae':averages,'development_folds':folds,
                      'test':{'start':test[0]['date'],'end':test[-1]['date'],'label_end':test[-1]['label_end_date'],
                              'n':len(test),'train_n':len(train),**test_metrics},'bundles':bundles,
                      'latest_fit_samples':len(rows),'latest_fit_feature_end':rows[-1]['date'],
                      'latest_fit_label_end':rows[-1]['label_end_date'],
                      'inference_as_of':live['date'],'latest_expected_return':RegressionPredictor(ROOT/bundles['latest']).predict(live)})
        entries.append(dict(code=code,horizon=1,**bundles))
        for r,estimate,mean in zip(test,estimated,mean_values):
            records.append({'code':code,'horizon':1,'date':r['date'],'label_end_date':r['label_end_date'],
                            'actual_return':r['target_return'],'expected_return':float(estimate),
                            'zero_return':0.,'historical_mean_return':float(mean)})
        for year in sorted({r['date'][:4] for r in test}):
            indices=[i for i,r in enumerate(test) if r['date'].startswith(year)]
            subset=[test[i] for i in indices]
            selected=return_metrics(subset,estimated[indices]);zero=return_metrics(subset,np.zeros(len(indices)))
            yearly.append({'code':code,'name':meta['name'],'origin_year':year,'n':len(indices),
                           'mae':selected['mae'],'rmse':selected['rmse'],'bias':selected['bias'],'zero_mae':zero['mae']})
        print(f'{code} {meta["name"]}: {winner}, MAE={test_metrics["selected"]["mae"]*100:.3f}pp; zero={test_metrics["zero"]["mae"]*100:.3f}pp',flush=True)
    for name,value in locked.items():assert sha(ROOT/name)==value,name
    assert sha(ROOT/'data/market.sqlite3')==db_sha
    aggregate={'tasks':len(tasks),'test_predictions':len(records),
               'mean_mae':float(np.mean([t['test']['selected']['mae'] for t in tasks])),
               'mean_zero_mae':float(np.mean([t['test']['zero']['mae'] for t in tasks])),
               'mae_better_than_zero':sum(t['test']['selected']['mae']<t['test']['zero']['mae']-1e-12 for t in tasks)}
    write(report/'summary.json',dict(provenance,protocol=p,tasks=tasks,aggregate=aggregate))
    write(report/'frozen_protocol.json',p)
    for filename,data in [('test_predictions.csv',records),('yearly_metrics.csv',yearly)]:
        with (report/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    lines=['# 下一交易日涨跌幅：训练与测试','',
           f'运行 `{run_id}`。{len(tasks)}个指数，收益目标为下一实际交易日收盘价/当日收盘价−1。数据截至{max(t["inference_as_of"] for t in tasks)}。','',
           '使用每个预测时点之前全部可用的已知标签训练样本。单个输入样本为过去252个交易日的15个价格特征，含新增的1/2/3/10日收益、5日波动；并非把全部原始价格直接塞进模型。',
           '', '2018—2019、2020—2021、2022—2023三个扩展训练验证窗口选型，所有训练标签结束日期严格早于验证/测试开始。测试从2024年起，A500从2025年起。候选为零收益、历史平均、标准化岭回归和浅层LightGBM；验证MAE改善不足1%时保留简单基线。测试不选型、不调参、不逐日重训。',
           '', '该价格区间此前已被其他期限实验观察，本次是探索性历史测试，不能宣称全新的独立未来验证。Latest只按已选种类用全部已知标签重新拟合，供未来使用，不能继承holdout成绩作为独立成绩。',
           '',f'{len(tasks)}组等权平均MAE：{aggregate["mean_mae"]*100:.3f}个百分点，零收益基准：{aggregate["mean_zero_mae"]*100:.3f}个百分点；{aggregate["mae_better_than_zero"]}/{len(tasks)}组优于零收益。小误差不等于具备稳定预测优势。','',
           '|指数|模型|训练样本（latest）|测试样本|MAE（百分点）|零收益MAE|RMSE（百分点）|当前预测|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for t in tasks:
        test=t['test'];selected=test['selected']
        lines.append(f'|{t["name"]}|{t["choice"]}|{t["latest_fit_samples"]}|{test["n"]}|{selected["mae"]*100:.3f}|{test["zero"]["mae"]*100:.3f}|{selected["rmse"]*100:.3f}|{t["latest_expected_return"]*100:+.4f}%|')
    lines+=['','收益为指数价格收益，不含基金分红、费用或跟踪误差。A500包含发布前回溯数据，部分行业历史较短。模型仍为研究用途。',
            '', '`summary.json`记录逐折选择、截止日期、数据/源码指纹和完整指标；`test_predictions.csv`可逐日比较，`yearly_metrics.csv`按预测起始年份分组。既有长周期模型、数据及旧报告的SHA256锁定并检查未变。','',
            '复验：`.venv/bin/python code/training/verify_one_day.py --run '+run_id+'`。','']
    (report/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    write(model_root/'index.json',{'run_id':run_id,'report':str((report/'summary.json').relative_to(ROOT)),
                                 'research_only':True,'models':entries})
    saved={str(path.relative_to(ROOT)):sha(path) for path in model_root.rglob('*') if path.is_file()}
    write(report/'verification.json',{'status':'passed','bundles_loaded':2*len(tasks),'test_predictions_replayed':len(records),
                                      'all_saved_file_sha256':saved,'original_artifact_sha256':locked})
    write(ROOT/'code/models/latest_one_day.json',{'run_id':run_id,'research_only':True,
              'index':str((model_root/'index.json').relative_to(ROOT)),
              'verification':str((report/'verification.json').relative_to(ROOT))})
    print('RUN_ID='+run_id,flush=True)


if __name__=='__main__':run()
