"""Date labels follow observed closes and each model's own prediction origin."""
import datetime as dt
import json
from pathlib import Path

CALENDAR = json.loads(Path(__file__).with_name('trading_calendar.json').read_text(encoding='utf-8'))
CLOSED = set(CALENDAR['closed_dates'])
CONFIRMED = set(CALENDAR['confirmed_years'])


def target_date(basis, horizon, observed):
    day = dt.date.fromisoformat(basis)
    last_observed = max(observed) if observed else basis
    estimated = False
    remaining = horizon
    while remaining:
        day += dt.timedelta(days=1)
        iso = day.isoformat()
        if iso <= last_observed:
            trading = iso in observed
        else:
            estimated |= day.year not in CONFIRMED
            trading = day.weekday() < 5 and iso not in CLOSED
        if trading:
            remaining -= 1
    return day.isoformat(), estimated


def annotate_dates(rows, horizons, calendar):
    observed = set(calendar)
    for row in rows:
        past = [d for d in calendar if d <= row['as_of']]
        row['date_ranges'] = {'actual': {}, 'forecast': {}}
        row['date_labels'] = {'actual': {}, 'forecast': {}}
        for h in horizons:
            key = str(h)
            start = past[-h-1] if len(past) > h and row['actual'][key] is not None else None
            row['date_ranges']['actual'][key] = {'start': start, 'end': row['as_of'], 'estimated': False}
            row['date_labels']['actual'][key] = ['从 '+start, '至 '+row['as_of']] if start else ['比较日期不足']
            basis = row.get('forecast_as_of', {}).get(key, row['as_of'])
            if row['forecast'][key] is None:
                basis = None
            end, estimated = target_date(basis, h, observed) if basis else (None, False)
            row['date_ranges']['forecast'][key] = {'start': basis, 'end': end, 'estimated': estimated}
            row['date_labels']['forecast'][key] = (['基于 '+basis, ('暂估至 ' if estimated else '至 ')+end]
                                                          if basis else ['预测日期未就绪'])
    headers = {}
    for kind in ('actual', 'forecast'):
        headers[kind] = {}
        for h in horizons:
            labels = [r['date_labels'][kind][str(h)] for r in rows]
            headers[kind][str(h)] = labels[0] if all(x == labels[0] for x in labels) else ['各指数日期不同', '详见各行']
    return headers
