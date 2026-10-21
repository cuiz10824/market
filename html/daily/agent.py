"""Persistent weekly collection, coverage audit and retry around saved-model inference."""
import datetime as dt
import json
import logging
from pathlib import Path
import threading

from daily.dates import CLOSED, CONFIRMED
from daily.feeds import SHANGHAI
from daily.page import atomic_text

LOG = logging.getLogger('market.agent')


def weekly_window(now):
    now = now.astimezone(SHANGHAI)
    monday = now.date()-dt.timedelta(days=now.weekday())
    due = dt.datetime.combine(monday, dt.time(9), SHANGHAI)
    if now < due:
        due -= dt.timedelta(days=7)
    return due, due.date()-dt.timedelta(days=7), due.date()-dt.timedelta(days=1)


def audit_week(service, start, end):
    dates = []
    unknown = set()
    day = start
    while day <= end:
        if day.year not in CONFIRMED:
            unknown.add(day.year)
        if day.weekday() < 5 and day.isoformat() not in CLOSED:
            dates.append(day.isoformat())
        day += dt.timedelta(days=1)
    # A completely closed week still needs a valid forecast from the preceding session.
    prior = end
    while prior.weekday() > 4 or prior.isoformat() in CLOSED:
        prior -= dt.timedelta(days=1)
    required = dates[-1] if dates else prior.isoformat()
    rows = {r['code']: r for r in service.payload['rows']}
    gaps = {}
    runtime = getattr(service, 'learned', None)
    for item in service.items:
        code = item['code']
        missing_prices = [d for d in dates if d not in service.prices[code]]
        bars = set(runtime.bars[code].date) if runtime else set()
        missing_bars = [d for d in dates if d not in bars]
        row = rows[code]
        missing_forecasts = [str(h) for h in service.payload['horizons']
            if row['forecast'].get(str(h)) is None or
            (row.get('forecast_as_of', {}).get(str(h)) or '') < required]
        if missing_prices or missing_bars or missing_forecasts or row['as_of'] < required:
            gaps[code] = dict(prices=missing_prices, ohlcv=missing_bars, forecast_horizons=missing_forecasts,
                              latest_close=row['as_of'])
    return dict(complete=not gaps and not unknown, start=start.isoformat(), end=end.isoformat(),
                expected_sessions=dates, required_close=required, gaps=gaps,
                calendar_warning=('需补充交易所休市表：'+','.join(map(str, sorted(unknown)))) if unknown else None)


class WeeklyAgent:
    def __init__(self, service, state_path=None, clock=None, enabled=True):
        self.service = service
        self.path = Path(state_path) if state_path else service.store.with_name('weekly_agent.json')
        self.clock = clock or (lambda: dt.datetime.now(SHANGHAI))
        self.lock = threading.Lock()
        self.enabled = enabled
        self.state = {'schema_version': 1, 'status': 'idle', 'completed_week': None,
                      'last_attempt': None, 'last_success': None, 'retry_after': None}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            if saved.get('schema_version') != 1:
                raise ValueError('Unsupported agent state')
            self.state.update(saved)
            if self.state['status'] == 'running':
                self.state.update(status='interrupted', retry_after=None)
        self.service.agent = self

    def snapshot(self):
        now = self.clock()
        due, start, end = weekly_window(now)
        # Read a shallow copy; state is replaced atomically rather than mutated by readers.
        state = dict(self.state)
        completed = state.get('completed_week') == end.isoformat()
        next_run = due+dt.timedelta(days=7) if completed else max(now, due)
        retry = state.get('retry_after')
        if not completed and retry and dt.datetime.fromisoformat(retry) > next_run:
            next_run = dt.datetime.fromisoformat(retry)
        if completed:
            last = state.get('last_attempt')
            daily = dt.datetime.fromisoformat(last)+dt.timedelta(minutes=30) if last else now
            next_run = min(next_run, max(now, daily))
        return dict(state, enabled=self.enabled, schedule='每周一09:00（北京时间）；错过后启动补跑',
                    target_week_start=start.isoformat(), target_week_end=end.isoformat(),
                    weekly_due=not completed, next_run=next_run.isoformat() if self.enabled else None,
                    poll_seconds=60, daily_refresh_minutes=30)

    def due(self, now):
        _, _, end = weekly_window(now)
        if self.state.get('completed_week') == end.isoformat():
            return False
        retry = self.state.get('retry_after')
        return not retry or now >= dt.datetime.fromisoformat(retry)

    def _save(self, **changes):
        state = dict(self.state, **changes)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(self.path, json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
        self.state = state

    def run(self, force=False, startup=False):
        if not self.lock.acquire(blocking=False):
            return False
        try:
            now = self.clock()
            weekly = self.due(now)
            last = self.state.get('last_attempt')
            daily = not last or now-dt.datetime.fromisoformat(last) >= dt.timedelta(minutes=30)
            if not (force or startup or (self.enabled and (weekly or daily))):
                return False
            _, start, end = weekly_window(now)
            week_pending = self.state.get('completed_week') != end.isoformat()
            self._save(status='running', phase='收集与校验8个指数日线，调用保存模型',
                       last_attempt=now.isoformat(), last_error=None)
            LOG.info('Refresh start: %s..%s', start, end)
            try:
                if not self.service.refresh():
                    self._save(status='waiting', retry_after=(now+dt.timedelta(minutes=5)).isoformat())
                    return False
                audit = audit_week(self.service, start, end)
                errors = self.service.last_refresh.get('errors', {})
                model_errors = (self.service.payload.get('model_selection') or {}).get('errors', {})
                success = audit['complete'] and not errors and not model_errors
                finished = self.clock()
                changes = dict(status='ready' if success else 'partial', phase='完成' if success else '等待补齐，保留可用预测',
                               coverage=audit, errors=errors, model_errors=model_errors,
                               retry_after=None if success else (finished+dt.timedelta(minutes=30)).isoformat(),
                               last_error=None)
                if success:
                    changes['last_success'] = finished.isoformat()
                    if week_pending:
                        changes['completed_week'] = end.isoformat()
                self._save(**changes)
                self.service.export_snapshot()
                LOG.info('Refresh %s; coverage=%s', self.state['status'], audit['complete'])
            except Exception as exc:
                self._save(status='error', phase='更新失败，保留已有数据', last_error=str(exc),
                           retry_after=(self.clock()+dt.timedelta(minutes=30)).isoformat())
                LOG.exception('Refresh failed')
                # Publish the failure status even when collection/inference failed.
                try:
                    self.service.export_snapshot()
                except Exception:
                    LOG.exception('Could not export failure status')
            return True
        finally:
            self.lock.release()
