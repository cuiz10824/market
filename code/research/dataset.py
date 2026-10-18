"""Price-only research samples and label-aware walk-forward splits. No ML dependencies."""
import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (5, 21, 63, 126)
LOOKBACK = 252
FEATURES = ('momentum_5', 'momentum_21', 'momentum_63', 'momentum_126',
            'momentum_252', 'volatility_21', 'volatility_63',
            'ma_gap_21', 'ma_gap_63', 'drawdown_252')


def load_history(code, root=ROOT):
    catalog = json.loads((root / 'data/catalog.json').read_text(encoding='utf-8'))['series']
    meta = next((m for m in catalog if m['code'] == code), None)
    if not meta or not meta.get('secid'):
        raise ValueError('Only the 14 primary index histories are eligible; community samples excluded')
    if meta['quality']['missing_observations_within_range']:
        raise ValueError('Primary history contains calendar gaps; review before use')
    calendar_path = root / 'data/sources/investment_data/2026-10-17/selected.zip'
    with zipfile.ZipFile(calendar_path) as archive:
        calendar = archive.read('calendars/day.txt').decode().splitlines()
    with sqlite3.connect(f'file:{root / "data/market.sqlite3"}?mode=ro', uri=True) as db:
        prices = dict(db.execute('SELECT date,close FROM prices WHERE code=? ORDER BY date', (code,)))
    return calendar, prices, meta


def aligned_prices(calendar, prices):
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError('Calendar must be nonempty, unique and increasing')
    unknown = set(prices) - set(calendar)
    if unknown:
        raise ValueError('Observations outside the reference calendar')
    values = [prices.get(day) for day in calendar]
    if any(v is not None and (not math.isfinite(v) or v <= 0) for v in values):
        raise ValueError('Invalid price')
    missing = [0]
    for value in values:
        missing.append(missing[-1] + (value is None))
    return values, missing


def sample_indices(calendar, prices, horizon):
    if horizon not in HORIZONS:
        raise ValueError('Supported horizons: 5, 21, 63, 126 trading sessions')
    values, missing = aligned_prices(calendar, prices)
    indices = [i for i in range(LOOKBACK, len(calendar) - horizon)
               if missing[i + horizon + 1] == missing[i - LOOKBACK]]
    return values, indices


def build_samples(calendar, prices, horizon, code=None):
    """Use closes known at t; target is close(t+h)/close(t)-1 on the calendar.

    A sample requires a continuous 252-session lookback AND complete forward
    interval. No filling, random splitting, pooled-index rows or fitted transforms.
    """
    values, indices = sample_indices(calendar, prices, horizon)
    rows = []
    for i in indices:
        close = values[i]
        row = {'date': calendar[i], 'label_end_date': calendar[i + horizon], 'horizon': horizon}
        if code is not None:
            row['code'] = code
        for h in (5, 21, 63, 126, 252):
            row[f'momentum_{h}'] = close / values[i - h] - 1
        for h in (21, 63):
            returns = [math.log(values[j] / values[j - 1]) for j in range(i - h + 1, i + 1)]
            row[f'volatility_{h}'] = statistics.stdev(returns) * math.sqrt(252)
            row[f'ma_gap_{h}'] = close / statistics.mean(values[i - h + 1:i + 1]) - 1
        row['drawdown_252'] = close / max(values[i - 251:i + 1]) - 1
        row['target_return'] = values[i + horizon] / close - 1
        row['target_up'] = int(row['target_return'] > 0)
        rows.append(row)
    return rows


def walk_forward(samples, min_train=504, test_size=126):
    """Expanding train window; purge every label not known BEFORE test begins.

    Dates, rather than row gaps, make the boundary safe after missing samples.
    Call separately per index and horizon. Hyperparameters need an inner split.
    """
    if min_train < 1 or test_size < 1:
        raise ValueError('Positive window sizes required')
    dates = [r['date'] for r in samples]
    if dates != sorted(set(dates)):
        raise ValueError('Use one unique, time-ordered index series per split')
    if len({r['horizon'] for r in samples}) > 1:
        raise ValueError('Use one forecast horizon per split')
    if len({r.get('code') for r in samples}) > 1:
        raise ValueError('Use one index code per split')
    if any(r['label_end_date'] <= r['date'] for r in samples):
        raise ValueError('Forward label must end after its observation')
    for start in range(min_train, len(samples), test_size):
        test = samples[start:start + test_size]
        train = [r for r in samples[:start] if r['label_end_date'] < test[0]['date']]
        if len(train) >= min_train:
            yield train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', default='000300')
    parser.add_argument('--horizon', type=int, choices=HORIZONS, default=21)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    calendar, prices, meta = load_history(args.code)
    samples = build_samples(calendar, prices, args.horizon, code=args.code)
    folds = list(walk_forward(samples))
    if args.output:
        if not samples:
            raise ValueError('No eligible samples')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
    print(json.dumps({'code': args.code, 'name': meta['name'], 'horizon': args.horizon,
                      'samples': len(samples), 'walk_forward_folds': len(folds),
                      'feature_cutoff': 'after daily close', 'status': 'exploratory_only',
                      'warnings': meta['quality']['warnings'],
                      'note': 'Features are price-only; no training, accuracy claim or trading PnL.'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
