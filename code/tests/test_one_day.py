import sys
import datetime as dt
import json
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'html'))
from daily.features import one_day_features as runtime_features
from research.one_day import FEATURES, build_one_day_samples, one_day_features
from training.engine import split_window
from test_research_dataset import history


class OneDayTests(unittest.TestCase):
    def test_all_single_day_candidates_save_reload_with_fifteen_features(self):
        import numpy as np
        from daily.regression import RegressionPredictor
        from training.one_day_models import fit, predict
        from training.engine import export_spec
        root=Path(__file__).resolve().parents[2]
        protocol=json.loads((root/'code/training/one_day_protocol.json').read_text(encoding='utf-8'))
        rng=np.random.default_rng(45)
        rows=[]
        for x in rng.normal(size=(350,len(FEATURES))):
            rows.append(dict(zip(FEATURES,x),target_return=float(.01*x[-1]+rng.normal(scale=.005))))
        for kind in protocol['regression_candidates']:
            fitted=fit(kind,rows,protocol)
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)
                bundle={'schema_version':1,'task':'next_session_return','code':'000300','horizon':1,
                        'features':list(FEATURES),'return':export_spec(fitted,folder,'return')}
                path=folder/'model.json';path.write_text(json.dumps(bundle),encoding='utf-8')
                loaded=RegressionPredictor(path)
                np.testing.assert_allclose([loaded.predict(r) for r in rows],predict(fitted,rows),rtol=1e-10,atol=1e-12)

    def test_label_is_next_exchange_session_not_next_calendar_day(self):
        calendar,prices=history(800)
        rows=build_one_day_samples(calendar,prices)
        self.assertEqual(len(rows),800-253)
        friday=next(r for r in rows if dt.date.fromisoformat(r['date']).weekday()==4)
        self.assertEqual(dt.date.fromisoformat(friday['label_end_date']).weekday(),0)
        self.assertAlmostEqual(friday['target_return'],prices[friday['label_end_date']]/prices[friday['date']]-1)
        self.assertEqual(rows[-1]['label_end_date'],calendar[-1])
        self.assertEqual(rows[-1]['date'],calendar[-2])

    def test_future_prices_cannot_change_current_features(self):
        calendar,prices=history(600)
        first=build_one_day_samples(calendar,prices)[0]
        changed={d:p*(2 if d>first['date'] else 1) for d,p in prices.items()}
        alternate=build_one_day_samples(calendar,changed)[0]
        self.assertEqual({f:first[f] for f in FEATURES},{f:alternate[f] for f in FEATURES})
        self.assertNotEqual(first['target_return'],alternate['target_return'])
        values=list(prices.values())
        self.assertEqual(one_day_features(values,252),runtime_features(values,252))

    def test_missing_day_is_not_silently_skipped_and_train_targets_are_purged(self):
        calendar,prices=history(1500)
        del prices[calendar[280]]
        rows=build_one_day_samples(calendar,prices)
        self.assertTrue(all(not (r['date']<=calendar[280]<=r['label_end_date']) for r in rows))
        train,test=split_window(rows,calendar[1000],min_train=100,min_test=50)
        self.assertLess(max(r['label_end_date'] for r in train),test[0]['date'])
        self.assertNotIn(calendar[-1],{r['date'] for r in rows})


if __name__=='__main__':unittest.main()
