"""Independent replay of all enriched native models; no fitting or test selection."""
import csv
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html')]
from daily.enhanced_predictor import EnhancedPredictor
from research.rich_dataset import load_bars
from daily.rich_features import build_frames
from training.train_enhanced import read,sha,write,metrics


def verify():
    pointer=read(ROOT/'code/models/latest_enhanced.json')
    summary=read(ROOT/pointer['report']);index=read(ROOT/pointer['index'])
    verification=read(ROOT/pointer['verification']);p=summary['protocol']
    for mapping in ('all_saved_file_sha256','original_artifact_sha256'):
        for name,digest in verification[mapping].items():assert sha(ROOT/name)==digest,name
    for name,digest in summary['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'data/enhanced/market_ohlcv.sqlite3')==summary['database_sha256']
    assert sha((ROOT/pointer['report']).parent/'frozen_protocol.json')==summary['protocol_sha256']
    bars,calendar=load_bars();codes=p['codes'];positions={d:i for i,d in enumerate(calendar)}
    features=build_frames(bars,codes,calendar)
    prices={c:bars[c].set_index('date').close.to_dict() for c in codes}
    with ((ROOT/pointer['report']).parent/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    groups={}
    for r in records:
        key=(int(r['horizon']),r['candidate'],'2025-01-01' if r['code']=='000510' else '2024-01-01')
        groups.setdefault(key,[]).append(r)
    replayed=0
    for entry in index['models']:
        model=EnhancedPredictor(ROOT/entry['bundle']);h=entry['horizon'];kind=entry['candidate']
        if entry['role']=='latest':
            data=[features[c].loc[calendar[-1]].to_dict() for c in codes]
            values=model.predict(data)
            for c,value in zip(codes,values):
                task=next(t for t in summary['tasks'] if t['code']==c and t['horizon']==h)
                assert abs(value-task['latest_candidate_returns'][kind])<1e-12
        else:
            group=groups[(h,kind,entry['start'])]
            assert model.bundle['metadata']['fit_label_end']<min(r['date'] for r in group)
            data=[]
            for r in group:
                assert positions[r['label_end_date']]-positions[r['date']]==h
                actual=prices[r['code']][r['label_end_date']]/prices[r['code']][r['date']]-1
                assert abs(actual-float(r['actual_return']))<1e-12
                data.append(features[r['code']].loc[r['date']].to_dict())
            values=model.predict(data)
            np.testing.assert_allclose(values,[float(r['expected_return']) for r in group],atol=1e-12,rtol=1e-10)
            for code in {r['code'] for r in group}:
                mask=np.array([r['code']==code for r in group])
                observed=np.array([float(r['actual_return']) for r in group])[mask]
                computed=metrics(observed,values[mask])
                task=next(t for t in summary['tasks'] if t['code']==code and t['horizon']==h)
                for name in ('mae_pp','error_std_pp','bias_pp','rmse_pp'):
                    assert abs(computed[name]-task['test_candidates'][kind][name])<1e-10
            replayed+=len(group)
    assert replayed==len(records)
    for task in summary['tasks']:
        means=task['validation_mean_mae_pp'];best=min(p['candidates'],key=lambda k:means[k])
        gain=1-means[best]/min(means[k] for k in ('zero','mean','legacy'))
        assert task['choice']==(best if gain>=p['min_relative_validation_gain'] else 'legacy')
    verification.update(independent_replay_passed=True,independent_predictions_replayed=replayed,
                        native_bundles_loaded=len(index['models']),fit_called_during_verification=False)
    write(ROOT/pointer['verification'],verification)
    print(f'PASS: {len(index["models"])} native bundles, {replayed} candidate predictions replayed; choices, labels, metrics and original artifacts verified.')


if __name__=='__main__':verify()
