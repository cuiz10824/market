import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import hashlib
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'html'))
from daily.agent import WeeklyAgent, audit_week, weekly_window
from daily.feeds import SHANGHAI
from daily.service import merge_checked
from daily.ohlcv_store import OHLCVStore


class FakeService:
    def __init__(self, folder, monday=dt.date(2026,10,12)):
        self.store=Path(folder)/'market.sqlite3'
        self.items=[{'code':'000300'},{'code':'000510'}]
        days=[(monday+dt.timedelta(days=i)).isoformat() for i in range(5)]
        self.prices={i['code']:{d:100. for d in days} for i in self.items}
        self.learned=SimpleNamespace(bars={i['code']:SimpleNamespace(date=list(days)) for i in self.items})
        self.payload={'horizons':[1,5],'rows':[dict(code=i['code'],as_of=days[-1],forecast={'1':1.,'5':2.},
            forecast_as_of={'1':days[-1],'5':days[-1]}) for i in self.items], 'model_selection':{'errors':{}}}
        self.last_refresh={'errors':{}}
        self.calls=0
        self.exports=0

    def refresh(self):
        self.calls+=1
        return True

    def export_snapshot(self):
        self.exports+=1


class AgentTests(unittest.TestCase):
    def test_failed_state_write_does_not_publish_fake_running_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent=WeeklyAgent(FakeService(tmp),clock=lambda:dt.datetime(2026,10,19,9,tzinfo=SHANGHAI))
            with patch('daily.agent.atomic_text',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):agent.run()
            self.assertEqual(agent.state['status'],'idle')
            self.assertIsNone(agent.state['last_attempt'])
            self.assertTrue(agent.lock.acquire(blocking=False))
            agent.lock.release()

    def test_monday_schedule_uses_shanghai_and_handles_before_nine(self):
        early=dt.datetime(2026,10,19,8,59,tzinfo=SHANGHAI)
        self.assertEqual(weekly_window(early)[2],dt.date(2026,10,11))
        due,start,end=weekly_window(early+dt.timedelta(minutes=1))
        self.assertEqual((start,end),(dt.date(2026,10,12),dt.date(2026,10,18)))
        self.assertEqual(due.hour,9)
        self.assertEqual(weekly_window(dt.datetime(2026,10,19,1,tzinfo=dt.timezone.utc)),(due,start,end))

    def test_success_persists_restarts_skip_duplicate_and_daily_refresh_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            now=[dt.datetime(2026,10,21,10,tzinfo=SHANGHAI)]
            service=FakeService(tmp)
            agent=WeeklyAgent(service,clock=lambda:now[0])
            self.assertTrue(agent.run())
            self.assertEqual(agent.state['status'],'ready')
            self.assertEqual(agent.state['completed_week'],'2026-10-18')
            restarted=WeeklyAgent(service,clock=lambda:now[0])
            self.assertFalse(restarted.run())
            now[0]+=dt.timedelta(minutes=30)
            self.assertTrue(restarted.run())
            self.assertEqual(service.calls,2)
            now[0]=dt.datetime(2026,10,28,10,tzinfo=SHANGHAI)
            self.assertTrue(restarted.due(now[0]))
            self.assertTrue(restarted.run())
            self.assertEqual(restarted.state['status'],'partial')
            self.assertEqual(restarted.state['completed_week'],'2026-10-18')

    def test_missing_ohlcv_or_stale_prediction_retries_then_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            now=[dt.datetime(2026,10,19,9,tzinfo=SHANGHAI)]
            service=FakeService(tmp)
            service.learned.bars['000300'].date.remove('2026-10-16')
            service.payload['rows'][0]['forecast_as_of']['1']='2026-10-15'
            agent=WeeklyAgent(service,clock=lambda:now[0])
            agent.run()
            self.assertEqual(agent.state['status'],'partial')
            self.assertIsNone(agent.state['completed_week'])
            self.assertIn('2026-10-16',agent.state['coverage']['gaps']['000300']['ohlcv'])
            self.assertFalse(agent.run())
            service.learned.bars['000300'].date.append('2026-10-16')
            service.payload['rows'][0]['forecast_as_of']['1']='2026-10-16'
            now[0]+=dt.timedelta(minutes=30)
            agent.run()
            self.assertEqual(agent.state['status'],'ready')

    def test_outage_does_not_mark_complete_and_crash_state_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=FakeService(tmp)
            service.last_refresh['errors']={'000300':'offline'}
            agent=WeeklyAgent(service,clock=lambda:dt.datetime(2026,10,19,9,tzinfo=SHANGHAI))
            agent.run()
            self.assertEqual(agent.state['status'],'partial')
            self.assertIsNone(agent.state['completed_week'])
            data=json.loads(agent.path.read_text());data['status']='running'
            agent.path.write_text(json.dumps(data))
            restarted=WeeklyAgent(service,clock=agent.clock)
            self.assertEqual(restarted.state['status'],'interrupted')
            self.assertTrue(restarted.due(agent.clock()))
            service.refresh=lambda:(_ for _ in ()).throw(ValueError('network down'))
            restarted.run()
            self.assertEqual(restarted.state['status'],'error')
            self.assertEqual(restarted.state['last_error'],'network down')

    def test_concurrent_triggers_only_collect_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=FakeService(tmp)
            entered=threading.Event();release=threading.Event()
            def fetch():
                service.calls+=1;entered.set();release.wait(5);return True
            service.refresh=fetch
            agent=WeeklyAgent(service,clock=lambda:dt.datetime(2026,10,19,9,tzinfo=SHANGHAI))
            thread=threading.Thread(target=agent.run);thread.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertFalse(agent.run(force=True))
                self.assertEqual(agent.snapshot()['status'],'running')
            finally:
                release.set();thread.join(5)
            self.assertEqual(service.calls,1)

    def test_missing_prices_can_be_backfilled_without_overwriting_history(self):
        calendar=[f'2026-{i//28+1:02d}-{i%28+1:02d}' for i in range(280)]
        old={d:100. for d in calendar};missing=calendar[-5];old.pop(missing)
        incoming={d:100. for d in calendar[-20:]}
        self.assertEqual(merge_checked(old,incoming,calendar)[missing],100.)

    def test_closed_week_and_unconfirmed_future_calendar(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=FakeService(tmp)
            for row in service.payload['rows']:
                row['as_of']='2026-02-13';row['forecast_as_of']={'1':'2026-02-13','5':'2026-02-13'}
            audit=audit_week(service,dt.date(2026,2,16),dt.date(2026,2,22))
            self.assertTrue(audit['complete'])
            self.assertEqual(audit['required_close'],'2026-02-13')
            unknown=audit_week(service,dt.date(2027,1,4),dt.date(2027,1,10))
            self.assertFalse(unknown['complete'])
            self.assertIn('2027',unknown['calendar_warning'])

    def test_ohlcv_backfill_repairs_internal_incremental_gap_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bars.sqlite3'
            digest=hashlib.sha256((ROOT/'data/enhanced/market_ohlcv.sqlite3').read_bytes()).hexdigest()
            store=OHLCVStore(ROOT,['000300'],digest,path)
            old=store.bars['000300'].tail(20).to_dict('records')
            last=old[-1]
            day19=dict(last,date='2026-10-19');day20=dict(last,date='2026-10-20')
            prices={r['date']:r['close'] for r in old+[day19,day20]}
            store.pending['000300']=old+[day20]
            store.commit({'000300':prices})
            self.assertNotIn('2026-10-19',set(store.bars['000300'].date))
            store.pending['000300']=old+[day19,day20]
            store.commit({'000300':prices})
            again=OHLCVStore(ROOT,['000300'],digest,path)
            self.assertEqual(again.bars['000300'].date.tail(2).tolist(),['2026-10-19','2026-10-20'])
            self.assertEqual(hashlib.sha256((ROOT/'data/enhanced/market_ohlcv.sqlite3').read_bytes()).hexdigest(),digest)
