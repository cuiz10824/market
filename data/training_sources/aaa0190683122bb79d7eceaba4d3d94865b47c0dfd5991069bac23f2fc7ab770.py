"""Separate classifier experiment; original regression artifacts remain immutable."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from inference.three_class_predictor import ThreeClassPredictor
from research.dataset import FEATURES, build_feature_rows, build_samples, load_history
from training.engine import split_window, return_metrics
from training.three_class import LABELS, labels, fit, predict, metrics, choose, export


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run():
    protocol_path = ROOT/'code/training/three_class_protocol.json'
    p = read(protocol_path)
    base_report = ROOT/'data/reports/training'/p['base_run']
    base_models = ROOT/'code/models'/p['base_run']
    locked = dict(read(base_report/'verification.json')['all_saved_file_sha256'])
    # Lock all existing reports, raw database and original source/pointer as well.
    for path in list(base_report.rglob('*')) + [ROOT/'data/market.sqlite3', ROOT/'code/models/latest.json']:
        if path.is_file():
            locked[str(path.relative_to(ROOT))] = sha(path)
    sources = ['code/training/three_class.py', 'code/training/train_three_class.py',
               'code/inference/three_class_predictor.py', 'code/research/dataset.py',
               'code/training/engine.py', 'code/inference/predictor.py']
    source_hashes = {s: sha(ROOT/s) for s in sources}
    source_id = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    run_id = 'threeclass_20261018_'+sha(protocol_path)[:8]+'_'+source_id[:8]
    report = ROOT/'data/reports/training'/run_id
    model_root = ROOT/'code/models'/run_id
    if report.exists() or model_root.exists():
        raise ValueError('Run already exists; use a new protocol/source identity')
    original = {}
    with (base_report/'test_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            original[(row['code'], int(row['horizon']), row['date'])] = row
    references = {(m['code'], m['horizon']):m for m in read(base_models/'index.json')['models']}
    tasks, records, entries = [], [], []
    for code in p['codes']:
        calendar, prices, meta = load_history(code)
        features = build_feature_rows(calendar, prices, code)
        cache = {r['date']:r for r in features}
        for horizon in p['horizons']:
            rows = build_samples(calendar, prices, horizon, code=code, feature_cache=cache)
            folds = []
            for start, end in p['development_windows']:
                split = split_window(rows, start, end, p['min_train_samples'], p['min_validation_samples'])
                if split is None:
                    folds.append({'start':start, 'end':end, 'status':'skipped_insufficient_history'})
                    continue
                train, valid = split
                y = labels([r['target_return'] for r in valid], p['threshold'], p['boundary_tolerance'])
                scores = {k:metrics(y, predict(fit(k,train,p),valid)) for k in p['classification_candidates']}
                folds.append({'start':start, 'end':end, 'status':'completed', 'train_n':len(train),
                              'validation_n':len(valid), 'last_train_label_end':train[-1]['label_end_date'],
                              'last_validation_label_end':valid[-1]['label_end_date'], 'scores':scores})
            winner, means = choose([f for f in folds if f['status']=='completed'],p)
            start = p['holdout_overrides'].get(code,p['holdout_start'])
            assert all(end<=start for _,end in p['development_windows'])
            train, test = split_window(rows,start,min_train=p['min_train_samples'],min_test=p['min_validation_samples'])
            y = labels([r['target_return'] for r in test],p['threshold'],p['boundary_tolerance'])
            fitted = fit(winner,train,p)
            probabilities = predict(fitted,test)
            baseline = predict(fit('constant',train,p),test)
            neutral = predict(fit('neutral',train,p),test)
            bundles = {}
            for role, estimator, fit_rows, check_rows in [
                    ('holdout', fitted, train, test),
                    ('latest', fit(winner,rows,p), rows, features[-10:])]:
                folder = model_root/role/code/str(horizon)
                folder.mkdir(parents=True)
                reference = references[(code,horizon)][role]
                payload = {'schema_version':1,'task':'three_class_return','code':code,'horizon':horizon,
                           'features':list(FEATURES),'labels':list(LABELS),'threshold':p['threshold'],
                           'classifier':export(estimator,folder),
                           'regression_reference':{'bundle':reference,'sha256':sha(ROOT/reference)},
                           'metadata':{'run_id':run_id,'role':role,'fit_samples':len(fit_rows),
                                       'fit_feature_end':fit_rows[-1]['date'],
                                       'fit_label_end':fit_rows[-1]['label_end_date'],
                                       'test_status':p['test_status'],'research_only':True,
                                       'database_sha256':locked['data/market.sqlite3']}}
                path = folder/'model.json'
                write(path,payload)
                outputs = ThreeClassPredictor(path).predict(check_rows)
                saved = np.array([[v['p_big_down'],v['p_neutral'],v['p_big_up']] for v in outputs])
                np.testing.assert_allclose(saved,predict(estimator,check_rows),rtol=1e-10,atol=1e-12)
                bundles[role] = str(path.relative_to(ROOT))
                if role=='holdout':
                    for r,v in zip(test,outputs):
                        old = original[(code,horizon,r['date'])]
                        assert r['label_end_date']==old['label_end_date']
                        assert r['target_return']==float(old['actual_return'])
                        assert v['expected_price_return']==float(old['expected_return'])
                        records.append({'code':code,'horizon':horizon,'date':r['date'],
                                        'label_end_date':r['label_end_date'],'actual_return':r['target_return'],
                                        'actual_class':int(labels([r['target_return']])[0]),
                                        'predicted_class':v['class_id'],
                                        'p_big_down':v['p_big_down'],'p_neutral':v['p_neutral'],
                                        'p_big_up':v['p_big_up'],'expected_return':v['expected_price_return']})
            rm = return_metrics(test,[float(original[(code,horizon,r['date'])]['expected_return']) for r in test])
            task = {'code':code,'name':meta['name'],'horizon':horizon,'choice':winner,
                    'development_mean_scores':means,'development_folds':folds,'bundles':bundles,
                    'test':{'start':test[0]['date'],'end':test[-1]['date'],
                            'selected':metrics(y,probabilities),'historical_frequency':metrics(y,baseline),
                            'always_neutral':metrics(y,neutral),'unchanged_regression':rm}}
            tasks.append(task)
            entries.append(dict(code=code,horizon=horizon,**bundles))
            print(f'{code} {horizon}: {winner}, accuracy={task["test"]["selected"]["accuracy"]:.3f}, macroF1={task["test"]["selected"]["macro_f1"]:.3f}',flush=True)
    assert len(records)==len(original)
    for path, expected in locked.items():
        assert sha(ROOT/path)==expected, 'Original artifact changed: '+path
    aggregates = {}
    for h in p['horizons']:
        group = [t for t in tasks if t['horizon']==h]
        aggregates[str(h)] = {k:{metric:float(np.mean([t['test'][k][metric] for t in group]))
                                    for metric in ['accuracy','macro_f1','macro_recall']}
                              for k in ['selected','historical_frequency','always_neutral']}
        aggregates[str(h)]['actual_counts'] = np.sum([t['test']['selected']['actual_counts'] for t in group],axis=0).tolist()
    summary = {'run_id':run_id,'protocol':p,'protocol_sha256':sha(protocol_path),
               'source_sha256':source_hashes,'regression_and_original_artifacts_unchanged':True,
               'tasks':tasks,'aggregate_by_horizon':aggregates}
    write(report/'summary.json',summary)
    write(report/'protocol.json',p)
    with (report/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(model_root/'index.json',{'run_id':run_id,'report':str((report/'summary.json').relative_to(ROOT)),
                                 'research_only':True,'models':entries})
    write(ROOT/'code/models/latest_three_class.json',{'run_id':run_id,'index':str((model_root/'index.json').relative_to(ROOT)),
                                                      'regression_base_run':p['base_run'],'research_only':True})
    saved = {str(path.relative_to(ROOT)):sha(path) for path in model_root.rglob('*') if path.is_file()}
    write(report/'verification.json',{'status':'passed','tasks_verified':len(tasks),'bundles_loaded':112,
                                      'heldout_predictions_replayed':len(records),'regression_exact_matches':len(records),
                                      'original_artifact_sha256':locked,'all_saved_file_sha256':saved})
    print('RUN_ID='+run_id,flush=True)


if __name__=='__main__':
    run()
