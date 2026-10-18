"""Replay saved three-class models and verify immutable regression without fitting."""
import csv
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from inference.three_class_predictor import ThreeClassPredictor
from research.dataset import load_history, build_feature_rows
from training.train_three_class import read, sha
from training.three_class import labels, metrics


def verify():
    pointer=read(ROOT/'code/models/latest_three_class.json')
    index=read(ROOT/pointer['index'])
    report=ROOT/index['report']
    summary=read(report)
    verification=read(report.parent/'verification.json')
    for mapping in ['original_artifact_sha256','all_saved_file_sha256']:
        for path,digest in verification[mapping].items():
            assert sha(ROOT/path)==digest,path
    for path,digest in summary['source_sha256'].items():
        assert sha(ROOT/path)==digest,path
    assert sha(ROOT/'code/training/three_class_protocol.json')==summary['protocol_sha256']
    with (report.parent/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    original=ROOT/'data/reports/training'/summary['protocol']['base_run']/'test_predictions.csv'
    with original.open(encoding='utf-8-sig',newline='') as f:
        old={(r['code'],r['horizon'],r['date']):r for r in csv.DictReader(f)}
    tasks={(t['code'],t['horizon']):t for t in summary['tasks']}
    features={}
    replayed=0
    for entry in index['models']:
        code,h=entry['code'],entry['horizon']
        if code not in features:
            calendar,prices,_=load_history(code)
            features[code]={r['date']:r for r in build_feature_rows(calendar,prices,code)}
        group=[r for r in records if r['code']==code and int(r['horizon'])==h]
        rows=[features[code][r['date']] for r in group]
        model=ThreeClassPredictor(ROOT/entry['holdout'])
        assert model.bundle['metadata']['fit_label_end']<group[0]['date']
        outputs=model.predict(rows)
        p=np.array([[o['p_big_down'],o['p_neutral'],o['p_big_up']] for o in outputs])
        saved=np.array([[float(r[k]) for k in ['p_big_down','p_neutral','p_big_up']] for r in group])
        np.testing.assert_allclose(p,saved,rtol=1e-12,atol=1e-12)
        y=labels([float(r['actual_return']) for r in group])
        assert y.tolist()==[int(r['actual_class']) for r in group]
        assert p.argmax(axis=1).tolist()==[int(r['predicted_class']) for r in group]
        computed=metrics(y,p)
        for key in ['accuracy','macro_f1','log_loss','brier']:
            assert abs(computed[key]-tasks[(code,h)]['test']['selected'][key])<1e-10
        for r,o in zip(group,outputs):
            baseline=old[(code,str(h),r['date'])]
            assert float(r['expected_return'])==o['expected_price_return']==float(baseline['expected_return'])
        ThreeClassPredictor(ROOT/entry['latest']).predict([list(features[code].values())[-1]])
        replayed+=len(group)
    assert replayed==len(records)==len(old)
    print(f'PASS: 112 bundles, {replayed} classification predictions replayed; original regression outputs and artifacts unchanged.')


if __name__=='__main__':verify()
