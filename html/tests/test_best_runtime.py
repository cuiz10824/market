import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'html'),str(ROOT/'code')]
from daily.service import DailyService
from daily.neural_features import pack as runtime_pack
from research.sequence_data import pack as training_pack
from daily.neural_predictor import NeuralPredictor
from daily.page import render_page


def read(path):return json.loads(path.read_text(encoding='utf-8'))


class BestRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.store=Path(cls.temp.name)
        cls.service=DailyService(store=cls.store/'daily.sqlite3')
        cls.selection=read(ROOT/'html/daily/model_selection.json')
        cls.summaries={k:read(ROOT/v['report']) for k,v in cls.selection['sources'].items()}

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_selection_is_validation_minimum_not_test_or_output_magnitude(self):
        self.assertEqual(len(self.selection['tasks']),40)
        for t in self.selection['tasks']:
            self.assertEqual(t['validation_mae_pp'],min(c['validation_mae_pp'] for c in t['candidate_scores']))
            self.assertNotIn(t['candidate'],('zero','mean'))
        self.assertFalse(self.selection['require_improvement_over_zero'])

    def test_forty_outputs_match_frozen_latest_candidates_without_fitting(self):
        payload=self.service.dashboard()
        self.assertEqual(payload['model_selection']['available_tasks'],40)
        self.assertEqual(payload['model_selection']['constant_baselines'],0)
        for row in payload['rows']:
            for h in (1,5,21,63,126):
                t=next(t for t in self.selection['tasks'] if t['code']==row['code'] and t['horizon']==h)
                source=self.summaries[t['source']]
                task=next(v for v in source['tasks'] if v['code']==row['code'] and v['horizon']==h)
                expected=task['latest_candidate_returns'][t['candidate']]*100
                self.assertAlmostEqual(row['forecast'][str(h)],expected,places=6)
                self.assertEqual(row['forecast_as_of'][str(h)],'2026-10-16')
                self.assertNotEqual(row['forecast'][str(h)],0.)
                self.assertNotIn('零变化',row['models'][str(h)])
        # Re-running daily inference cannot call a training adapter.
        with patch('training.neural_engine.fit',side_effect=AssertionError('Daily must not fit')):
            again=self.service.build_payload()
        self.assertEqual(again['rows'],payload['rows'])

    def test_daily_pack_has_exact_training_features_and_weight_tampering_fails(self):
        calendar=self.service.calendar
        raw=self.service.learned.bars['000300'].set_index('date').reindex(calendar)[['open','high','low','close','volume']].to_numpy(dtype=float)
        for length in (252,756,1260):
            ends=np.array([len(calendar)-1])
            np.testing.assert_array_equal(training_pack(raw,ends,length),runtime_pack(raw,ends,length))
        t=next(t for t in self.selection['tasks'] if t['profile']=='neural')
        original=ROOT/t['bundles'][0];bundle=read(original)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'model.json').write_text(json.dumps(bundle));(folder/bundle['weights']['file']).write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError,'checksum'):NeuralPredictor(folder/'model.json')

    def test_candle_catchup_starts_from_candle_cache_not_newer_close_only_date(self):
        runtime=self.service.learned
        item=next(i for i in self.service.items if i['code']=='000300')
        candles=runtime.bars['000300'].tail(20).to_dict('records')
        with patch('daily.ohlcv_store.fetch_bars',return_value=candles) as fetch:
            runtime.fetch(item,'2026-12-01')
        fetch.assert_called_once_with(item,'2026-10-16')
        runtime.pending.pop('000300',None)

    def test_new_prices_without_ohlcv_keep_previous_basis_and_missing_history_never_fills_zero(self):
        base=self.service
        def fake(item,last_date):
            values=base.prices[item['code']]
            result={d:values[d] for d in sorted(values)[-20:]}
            result['2026-10-19']=values[max(values)]*1.01
            return result,'fixture close-only'
        with tempfile.TemporaryDirectory() as temp:
            s=DailyService(store=Path(temp)/'daily.sqlite3',fetcher=fake);s.refresh()
            self.assertEqual(s.payload['model_selection']['stale_tasks'],40)
            for r in s.payload['rows']:
                self.assertEqual(r['as_of'],'2026-10-19')
                self.assertTrue(all(d=='2026-10-16' for d in r['forecast_as_of'].values()))
            page=render_page(ROOT,s.payload)
            self.assertIn('基于 2026-10-16',page)
            # A missing old observation invalidates the relevant model input;
            # there is no fabricated zero or substitute baseline.
            f=s.learned.bars['000300'];s.learned.bars['000300']=f.drop(f.index[-200])
            failed=s.build_payload();row=next(r for r in failed['rows'] if r['code']=='000300')
            self.assertTrue(all(v is None for v in row['forecast'].values()))
            self.assertTrue(row['forecast_errors'])

    def test_valid_new_ohlcv_updates_predictions_survives_restart_and_rejects_unit_changes(self):
        holder={}
        def fake(item,last_date):
            s=holder['service'];code=item['code'];bars=s.learned.bars[code].tail(20).to_dict('records')
            last=bars[-1];price=last['close']*1.01
            bars.append(dict(date='2026-10-19',open=last['close'],high=price*1.01,low=last['close']*.995,close=price,volume=last['volume']))
            with s.learned.lock:s.learned.pending[code]=bars
            return {r['date']:r['close'] for r in bars},'fixture OHLCV'
        with tempfile.TemporaryDirectory() as temp:
            store=Path(temp)/'daily.sqlite3';s=DailyService(store=store,fetcher=fake);holder['service']=s
            before=[v for r in s.payload['rows'] for v in r['forecast'].values()]
            s.refresh()
            self.assertEqual(s.payload['model_selection']['stale_tasks'],0)
            self.assertEqual(s.payload['model_selection']['available_tasks'],40)
            self.assertTrue(all(all(d=='2026-10-19' for d in r['forecast_as_of'].values()) for r in s.payload['rows']))
            after=[v for r in s.payload['rows'] for v in r['forecast'].values()]
            self.assertFalse(np.allclose(before,after))
            restarted=DailyService(store=store)
            self.assertEqual([r['forecast'] for r in restarted.payload['rows']],[r['forecast'] for r in s.payload['rows']])
            bad=restarted.learned.bars['000300'].tail(20).to_dict('records')
            for r in bad:r['volume']*=10
            last=dict(bad[-1],date='2026-10-20');bad.append(last)
            restarted.learned.pending['000300']=bad
            accepted=dict(restarted.prices['000300'],**{'2026-10-20':last['close']})
            restarted.learned.commit({'000300':accepted})
            self.assertIn('重叠校验失败',restarted.learned.errors['000300'])
            self.assertEqual(restarted.learned.bars['000300'].date.max(),'2026-10-19')


if __name__=='__main__':unittest.main()
