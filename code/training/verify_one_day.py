"""Replay saved next-session regressors and labels; never retrain or select."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'html'))
from daily.regression import RegressionPredictor
from research.dataset import load_history
from research.one_day import build_one_day_samples, one_day_features
from training.engine import return_metrics
from training.train_one_day import read, sha
from training.provenance import historical_source


def verify(run_id=None):
    if run_id is None:run_id=read(ROOT/'code/models/latest_one_day.json')['run_id']
    if Path(run_id).name!=run_id:raise ValueError('Invalid run ID')
    report=ROOT/'data/reports/training'/run_id
    summary=read(report/'summary.json');verification=read(report/'verification.json')
    for mapping in ('all_saved_file_sha256','original_artifact_sha256'):
        for path,digest in verification[mapping].items():assert sha(ROOT/path)==digest,path
    assert sha(ROOT/'data/market.sqlite3')==summary['database_sha256']
    assert sha(report/'frozen_protocol.json')==summary['protocol_sha256']
    for path,digest in summary['source_sha256'].items():historical_source(ROOT,path,digest)
    with (report/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    count=0
    for task in summary['tasks']:
        code=task['code'];calendar,prices,_=load_history(code)
        features={r['date']:r for r in build_one_day_samples(calendar,prices,code)}
        group=[r for r in records if r['code']==code]
        model=RegressionPredictor(ROOT/task['bundles']['holdout'])
        assert model.bundle['metadata']['fit_label_end']<group[0]['date']
        estimates=[];rows=[]
        for r in group:
            row=features[r['date']]
            assert row['label_end_date']==r['label_end_date']
            assert calendar.index(r['label_end_date'])-calendar.index(r['date'])==1
            assert row['target_return']==float(r['actual_return'])
            prediction=model.predict(row)
            assert abs(prediction-float(r['expected_return']))<1e-12
            estimates.append(prediction);rows.append(row)
        metrics=return_metrics(rows,estimates)
        for key in ('mae','rmse','bias'):
            assert abs(metrics[key]-task['test']['selected'][key])<1e-12
        latest=RegressionPredictor(ROOT/task['bundles']['latest'])
        live=one_day_features([prices[d] for d in calendar[-253:]],252)
        assert abs(latest.predict(live)-task['latest_expected_return'])<1e-12
        assert latest.bundle['metadata']['fit_label_end']<=calendar[-1]
        count+=len(group)
    assert count==len(records)==summary['aggregate']['test_predictions']
    print(json.dumps({'status':'passed','tasks':len(summary['tasks']),'bundles_loaded':2*len(summary['tasks']),
                      'predictions_replayed':count,'original_horizons_unchanged':True,
                      'fit_called_during_verification':False},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run');args=parser.parse_args()
    verify(args.run)
