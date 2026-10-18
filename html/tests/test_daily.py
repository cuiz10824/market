import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'html'))
sys.path.insert(0,str(ROOT/'code'))
from daily.features import features_at
from daily.feeds import completed, parse_tencent, SHANGHAI
from daily.service import DailyService, merge_checked
from inference.predictor import Predictor
from research.dataset import features_at as training_features
from server import handler_for


class DailyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.service=DailyService(store=Path(cls.tmp.name)/'base.sqlite3')

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_56_regression_outputs_exactly_match_original_saved_models(self):
        service=self.service
        for item in service.items:
            prices=service.prices[item['code']]
            dates=[d for d in service.calendar if d<=max(prices)][-253:]
            values=[prices[d] for d in dates]
            feature=features_at(values,252)
            self.assertEqual(feature,training_features(values,252))
            row=next(r for r in service.payload['rows'] if r['code']==item['code'])
            for h in (5,21,63,126):
                model=service.models[(item['code'],h)]
                reference=ROOT/'code/models'/service.run_id/'latest'/item['code']/str(h)/'model.json'
                expected=Predictor(reference).predict([feature])[0]['expected_price_return']
                self.assertEqual(model.predict(feature),expected)
                self.assertEqual(row['forecast'][str(h)],expected*100)
            self.assertIsNone(row['forecast']['1'])
            self.assertNotIn('p_up',row)

    def test_feed_rejects_wrong_identity_and_partial_daily_candles(self):
        item={'code':'000300','name':'沪深300','secid':'1.000300'}
        now=dt.datetime(2026,10,16,14,tzinfo=SHANGHAI)
        candles=[['2026-10-15','100','101','102','99'],['2026-10-16','101','102','103','100']]
        self.assertEqual(completed(candles,now),{'2026-10-15':101})
        payload={'code':0,'data':{'sh000300':{'day':candles,'qt':{'sh000300':['','错误指数','000300']}}}}
        with self.assertRaises(ValueError):parse_tencent(payload,item,now)
        for candle in [['2026-10-19','100','101','102','99'],['2026-10-16','100','101','100','99'],['2026-10-16','100','nan','102','99']]:
            with self.assertRaises(ValueError):completed([candle],now)

    def test_gaps_revisions_and_unverified_dates_are_rejected(self):
        old=self.service.prices['000300'];calendar=self.service.calendar
        dates=sorted(old)[-20:];new={d:old[d] for d in dates}
        self.assertEqual(merge_checked(old,new,calendar),old)
        changed=dict(new);changed[dates[-1]]+=1
        with self.assertRaises(ValueError):merge_checked(old,changed,calendar)
        with self.assertRaises(ValueError):merge_checked(old,{dates[-1]:old[dates[-1]]},calendar)
        new['2099-01-01']=100
        with self.assertRaises(ValueError):merge_checked(old,new,calendar)

    def test_incremental_refresh_restart_and_partial_failure_preserve_base(self):
        digest=hashlib.sha256((ROOT/'data/market.sqlite3').read_bytes()).hexdigest()
        base=self.service
        # Simulate later exchange closes. Fixture prices stay inside a temporary DB.
        def fake_fetch(item,last_date):
            code=item['code']
            if code=='399989':raise ValueError('simulated outage')
            prices=base.prices[code]
            result={d:prices[d] for d in sorted(prices)[-20:]}
            result['2026-10-19']=prices[max(prices)]*1.01
            result['2026-10-20']=prices[max(prices)]*1.02
            return result,'test fixture'
        with tempfile.TemporaryDirectory() as tmp:
            store=Path(tmp)/'daily.sqlite3'
            service=DailyService(store=store,fetcher=fake_fetch)
            service.refresh()
            medical=next(r for r in service.payload['rows'] if r['code']=='399989')
            self.assertEqual(medical['as_of'],'2026-10-16')
            self.assertTrue(medical['update_error'])
            current=next(r for r in service.payload['rows'] if r['code']=='000300')
            self.assertEqual(current['as_of'],'2026-10-20')
            self.assertAlmostEqual(current['actual']['1'],(1.02/1.01-1)*100)
            restarted=DailyService(store=store)
            self.assertEqual(restarted.prices['000300'],service.prices['000300'])
            with sqlite3.connect(store) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM prices').fetchone()[0],26)
        self.assertEqual(hashlib.sha256((ROOT/'data/market.sqlite3').read_bytes()).hexdigest(),digest)

    def test_anchor_failure_rejects_unverified_new_dates(self):
        base=self.service
        def fake_fetch(item,last_date):
            if item['code']=='000300':raise ValueError('anchor unavailable')
            old=base.prices[item['code']]
            return dict({d:old[d] for d in sorted(old)[-20:]},**{'2026-10-19':old[max(old)]}),'fixture'
        with tempfile.TemporaryDirectory() as tmp:
            service=DailyService(store=Path(tmp)/'daily.sqlite3',fetcher=fake_fetch)
            service.refresh()
            self.assertEqual(service.last_refresh['successful_indices'],0)
            self.assertTrue(all(r['as_of']=='2026-10-16' for r in service.payload['rows']))

    def test_snapshot_export_is_valid_and_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=DailyService(store=Path(tmp)/'daily.sqlite3')
            # Export into a temporary website, retaining real model results.
            service.root=Path(tmp)
            (service.root/'html').mkdir()
            service.export_snapshot()
            service.export_snapshot()
            path=service.root/'html/snapshot.js'
            raw=path.read_text(encoding='utf-8')
            payload=json.loads(raw.removeprefix('window.MARKET_SNAPSHOT = ').strip().removesuffix(';'))
            self.assertEqual(len(payload['rows']),14)
            self.assertEqual(list((service.root/'html').iterdir()),[path])

    def test_http_dashboard_and_private_paths(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(self.service))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            base=f'http://127.0.0.1:{server.server_port}'
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(base+'/api/dashboard') as response:
                self.assertEqual(len(json.load(response)['rows']),14)
            for path in ['/server.py','/daily/service.py','/../data/market.sqlite3']:
                with self.assertRaises(urllib.error.HTTPError) as error:opener.open(base+path)
                self.assertEqual(error.exception.code,404)
            request=urllib.request.Request(base+'/api/refresh',data=b'',headers={'Origin':'https://example.com'})
            with self.assertRaises(urllib.error.HTTPError) as error:opener.open(request)
            self.assertEqual(error.exception.code,403)
        finally:server.shutdown();server.server_close()


if __name__=='__main__':unittest.main()
