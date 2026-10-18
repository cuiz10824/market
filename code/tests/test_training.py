import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.predictor import Predictor
from research.dataset import FEATURES
from training.engine import (choose, direction_metrics, fit, predict, return_metrics,
                             save_bundle, split_window)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((Path(__file__).resolve().parents[1]/'training/protocol.json').read_text())
        rng = np.random.default_rng(42)
        self.rows = []
        for i, values in enumerate(rng.normal(size=(350,len(FEATURES)))):
            row = dict(zip(FEATURES,values))
            row.update(target_up=int(values[0]+.1*values[1]>0), target_return=float(.05*values[0]))
            self.rows.append(row)

    def test_all_saved_estimator_kinds_reload_with_equal_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            for direction in self.protocol['classification_candidates']:
                for regression in self.protocol['regression_candidates']:
                    clf = fit(direction,'direction',self.rows,self.protocol)
                    reg = fit(regression,'return',self.rows,self.protocol)
                    path = save_bundle(Path(temp)/f'{direction}-{regression}','000300',21,clf,reg,{})
                    result = Predictor(path).predict(self.rows[:8])
                    np.testing.assert_allclose([r['p_up'] for r in result],predict(clf,'direction',self.rows[:8]),rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose([r['expected_price_return'] for r in result],predict(reg,'return',self.rows[:8]),rtol=1e-10,atol=1e-12)

    def test_label_dates_are_purged_and_validation_targets_stay_inside_window(self):
        rows = [dict(date='2022-01-01',label_end_date='2023-12-29'),
                dict(date='2023-11-01',label_end_date='2024-01-02'),
                dict(date='2024-01-02',label_end_date='2024-06-01'),
                dict(date='2024-10-01',label_end_date='2025-04-01')]
        train, valid = split_window(rows,'2024-01-01','2025-01-01',1,1)
        self.assertEqual(train,[rows[0]])
        self.assertEqual(valid,[rows[2]])

    def test_small_validation_gain_retains_simpler_baseline(self):
        chosen,_ = choose({'constant':[.24],'momentum':[.25],'logistic':[.2399],'lightgbm':[.241]},'direction',self.protocol)
        self.assertEqual(chosen,'constant')
        chosen,_ = choose({'zero':[.10],'mean':[.11],'ridge':[.0995],'lightgbm':[.12]},'return',self.protocol)
        self.assertEqual(chosen,'zero')

    def test_scores_match_known_values_and_single_class_auc_is_explicit(self):
        rows=[dict(date='2020-01-01',label_end_date='2020-01-08',target_up=1,target_return=.1),
              dict(date='2020-01-08',label_end_date='2020-01-15',target_up=1,target_return=.2)]
        score=direction_metrics(rows,[.5,.5],5)
        self.assertEqual(score['brier'],.25)
        self.assertEqual(score['nonoverlap_n'],2)
        self.assertIsNone(score['auc'])
        self.assertAlmostEqual(return_metrics(rows,[0,0])['mae'],.15)

    def test_changed_booster_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            clf=fit('lightgbm','direction',self.rows,self.protocol)
            reg=fit('zero','return',self.rows,self.protocol)
            path=save_bundle(Path(temp),'000300',21,clf,reg,{})
            booster=path.parent/'direction.lgb.txt'
            booster.write_text(booster.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'checksum'):
                Predictor(path)


if __name__=='__main__':
    unittest.main()
