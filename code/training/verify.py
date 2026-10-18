"""Independent replay of saved models, labels and final-test scores; never fits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from inference.predictor import Predictor
from research.dataset import build_feature_rows, load_history
from training.engine import direction_metrics, return_metrics


def verify(run_id, register=False):
    if Path(run_id).name != run_id:
        raise ValueError('Invalid run ID')
    model_root = ROOT / 'code/models' / run_id
    output = ROOT / 'data/reports/training' / run_id
    index = json.loads((model_root / 'index.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    protocol = json.loads((output / 'frozen_protocol.json').read_text())
    expected_protocol = hashlib.sha256((output / 'frozen_protocol.json').read_bytes()).hexdigest()
    if expected_protocol != summary['protocol_sha256']:
        raise ValueError('Frozen protocol hash differs')
    if hashlib.sha256((ROOT/'data/market.sqlite3').read_bytes()).hexdigest() != summary['database_sha256']:
        raise ValueError('Database changed; replay requires exact training snapshot')
    for source, fingerprint in summary['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest() != fingerprint:
            raise ValueError('Source changed since training: '+source)
    if index['run_id'] != run_id or summary['run_id'] != run_id:
        raise ValueError('Run IDs differ')
    with (output/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
        stored = list(csv.DictReader(stream))
    groups = {}
    for row in stored:
        groups.setdefault((row['code'],int(row['horizon'])),[]).append(row)
    indexed = {(m['code'],m['horizon']):m for m in index['models']}
    recorded = {(m['code'],m['horizon']):m for m in summary['tasks']}
    if len(indexed) != len(index['models']) or set(indexed) != set(recorded) or set(groups) != set(recorded):
        raise ValueError('Missing or duplicate task records')
    caches = {}
    for code in {pair[0] for pair in groups}:
        calendar,prices,_ = load_history(code)
        caches[code] = (prices,{r['date']:r for r in build_feature_rows(calendar,prices,code=code)},
                        {day:i for i,day in enumerate(calendar)})
    n = 0
    file_manifest = {}
    for pair in sorted(groups):
        code,horizon = pair
        result = recorded[pair]
        prices,features,calendar_positions = caches[code]
        rows = groups[pair]
        if len(rows) != result['test']['selected_direction']['n']:
            raise ValueError('Prediction row count mismatch')
        dates = [r['date'] for r in rows]
        if dates != sorted(set(dates)):
            raise ValueError('Prediction dates duplicated or disordered')
        test_features = [features[r['date']] for r in rows]
        label_rows = []
        for row in rows:
            if calendar_positions[row['label_end_date']] - calendar_positions[row['date']] != horizon:
                raise ValueError('Stored label horizon differs from trading calendar')
            actual = prices[row['label_end_date']]/prices[row['date']]-1
            if abs(actual-float(row['actual_return']))>1e-12 or int(actual>0)!=int(row['actual_up']):
                raise ValueError('Stored test target differs from database')
            label_rows.append(dict(date=row['date'],label_end_date=row['label_end_date'],target_up=int(actual>0),target_return=actual))
        for role,key in [('holdout','holdout'),('latest','latest')]:
            path = ROOT/indexed[pair][key]
            if not path.resolve().is_relative_to(model_root.resolve()):
                raise ValueError('Bundle escaped run directory')
            predictor = Predictor(path)
            if predictor.bundle['code'] != code or predictor.bundle['horizon'] != horizon:
                raise ValueError('Bundle identity mismatch')
            meta = predictor.bundle['metadata']
            if meta['run_id'] != run_id or meta['database_sha256'] != summary['database_sha256']:
                raise ValueError('Bundle provenance mismatch')
            if role == 'holdout':
                if meta['fit_label_end'] >= rows[0]['date']:
                    raise ValueError('Final-test model leaked future labels')
                replayed = predictor.predict(test_features)
                pp = [r['p_up'] for r in replayed]
                rr = [r['expected_price_return'] for r in replayed]
                np.testing.assert_allclose(pp,[float(r['p_up']) for r in rows],rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(rr,[float(r['expected_return']) for r in rows],rtol=1e-10,atol=1e-12)
                dm,rm = direction_metrics(label_rows,pp,horizon),return_metrics(label_rows,rr)
                for metric in ('brier','accuracy','log_loss','nonoverlap_brier'):
                    if abs(dm[metric]-result['test']['selected_direction'][metric])>1e-12:
                        raise ValueError('Reported classification score differs')
                for metric in ('mae','rmse','bias'):
                    if abs(rm[metric]-result['test']['selected_return'][metric])>1e-12:
                        raise ValueError('Reported regression score differs')
            else:
                current = predictor.predict([features[max(features)]])[0]
                if not 0 <= current['p_up'] <= 1:
                    raise ValueError('Invalid latest prediction')
            n += 1
            for artifact in path.parent.iterdir():
                if artifact.is_file():
                    file_manifest[artifact.relative_to(ROOT).as_posix()] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    verification = {'run_id':run_id,'status':'passed','tasks_verified':len(groups),'bundles_loaded':n,
                    'heldout_predictions_replayed':len(stored),'all_saved_file_sha256':file_manifest,
                    'fit_called_during_verification':False,'stable_predictive_edge_demonstrated':False,
                    'note':'Technical replay passed. This is not certification of investment performance.'}
    (output/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    if register:
        expected = {(code,h) for code in protocol['codes'] for h in protocol['horizons']}
        if set(groups) != expected:
            raise ValueError('Only a complete run can become the default research model set')
        pointer = {'run_id':run_id,'research_only':True,'stable_predictive_edge_demonstrated':False,
                   'index':(model_root/'index.json').relative_to(ROOT).as_posix(),
                   'report':(output/'README.md').relative_to(ROOT).as_posix(),
                   'verification':(output/'verification.json').relative_to(ROOT).as_posix()}
        (ROOT/'code/models/latest.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k!='all_saved_file_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--register',action='store_true')
    args=parser.parse_args()
    verify(args.run,args.register)
