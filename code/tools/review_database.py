"""Reproducible readiness review; complements file-integrity audit with modeling limits."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from research.dataset import HORIZONS, load_history, sample_indices
from tools import market_data


def review(as_of):
    market_data.audit()
    catalog = json.loads((ROOT / 'data/catalog.json').read_text(encoding='utf-8'))['series']
    primary = []
    for meta in catalog:
        if not meta.get('secid'):
            continue
        calendar, prices, _ = load_history(meta['code'])
        sessions = sorted(prices)
        daily = [(sessions[i], prices[sessions[i]] / prices[sessions[i - 1]] - 1)
                 for i in range(1, len(sessions))]
        primary.append({'code': meta['code'], 'name': meta['name'], 'rows': len(prices),
                        'start': sessions[0], 'end': sessions[-1],
                        'calendar_gaps': meta['quality']['missing_observations_within_range'],
                        'max_absolute_daily_return': max(abs(r) for _, r in daily),
                        'large_moves_over_15_percent': [{'date': d, 'return': r} for d, r in daily if abs(r) > .15],
                        'eligible_samples_after_252_session_lookback': {
                            str(h): len(sample_indices(calendar, prices, h)[1]) for h in HORIZONS},
                        'formal_backtest_ready': False,
                        'note': 'Release/backfill and point-in-time history not audited; price-only exploratory use.'})
    community = [{'code': m['code'], 'name': m['name'], 'start': m['start'], 'end': m['end'],
                  'calendar_gaps': m['quality']['missing_observations_within_range'],
                  'baseline_training_selection': 'excluded'}
                 for m in catalog if not m.get('secid')]
    probe_path = ROOT / 'data/reports' / f'latest_probe_{as_of}.json'
    live = json.loads(probe_path.read_text(encoding='utf-8')) if probe_path.exists() else None
    primary_latest_confirmed = bool(live) and len(live['results']) == len(primary) and (
        {r['code'] for r in live['results']} == {r['code'] for r in primary}) and all(
        r['status'] == 'checked' and r['same_latest_date'] and r['close_difference'] <= .010001
        for r in live['results'])
    # Agreement of two vendor snapshots is a consistency test, not proof of truth.
    hs300, _ = market_data.load_series(catalog[0])
    raw = json.loads((ROOT / 'data/sources/tencent/000300.json').read_text(encoding='utf-8'))
    alternate = market_data.parse_tencent(raw, catalog[0])
    price_map = {r['date']: float(r['close']) for r in hs300}
    differences = [abs(float(r['close']) - price_map[r['date']]) for r in alternate if r['date'] in price_map]
    return {'as_of': as_of, 'database_sha256': market_data.sha256(ROOT / 'data/market.sqlite3'),
            'summary': {'series': len(catalog), 'rows': sum(m['rows'] for m in catalog),
                        'primary_series': len(primary), 'primary_price_coverage_passed': all(m['calendar_gaps'] == 0 for m in primary),
                        'primary_latest_live_probe_passed': primary_latest_confirmed,
                        'community_series_excluded_from_training': len(community),
                        'complete_training_database': False,
                        'verdict': 'Price history supports exploratory baselines; full training database remains incomplete.'},
            'primary': primary, 'community': community,
            'hs300_vendor_consistency': {'compared_closes': len(differences),
                                        'max_absolute_difference_points': round(max(differences), 8),
                                        'differences_over_half_cent': sum(d > .005 for d in differences),
                                        'independent_origin_confirmed': False},
            'latest_probe_report': probe_path.relative_to(ROOT).as_posix() if live else None,
            'missing_or_unaudited': [
                'Official index launch dates, pre-launch backfill and historical methodology changes',
                'Independent full-history price verification and authoritative holiday-calendar reconciliation',
                'Unified OHLCV/turnover features (available in raw responses, not yet standardized)',
                'Historical PE/PB/dividend yield, macro data and their actual publication timestamps',
                'Total-return indices and ETF/fund NAV, distributions, fees and tracking error',
                'Recent complete Shenwan industry histories and classification-version mappings'],
            'leakage_controls_required': [
                'Separate 5/21/63/126 trading-session labels and per-index evaluation',
                'Purge training labels whose end is on or after the test boundary',
                'Fit features, scaling, regime parameters and tuning within training windows only',
                'Use forward-filtered state probabilities; never smoothed historical probabilities',
                'Reserve a final untouched time interval; compare to simple baseline probabilities']}


def markdown(report):
    s = report['summary']
    cross = report['hs300_vendor_consistency']
    lines = [f"# 数据库复查 · {report['as_of']}", '',
             f"现有 {s['series']} 条序列、{s['rows']:,} 条收盘记录。14条主要指数可以用于价格型探索基线，**尚不能称为完整训练数据库**。", '',
             f"本次14条主要指数最新观测联网复核：{'全部通过' if s['primary_latest_live_probe_passed'] else '未全部确认'}。", '',
             '| 指数 | 代码 | 起止日期 | 日记录 | 缺日 | 半年标签样本数（252日预热后） |',
             '| --- | --- | --- | ---: | ---: | ---: |']
    for m in report['primary']:
        lines.append(f"| {m['name']} | {m['code']} | {m['start']} ~ {m['end']} | {m['rows']} | {m['calendar_gaps']} | {m['eligible_samples_after_252_session_lookback']['126']} |")
    lines += ['', '样本数是可构造标签的记录数，不是独立样本数；半年标签高度重叠。最后126个交易日尚无半年结果，不能伪造标签。', '',
              '## 数据限制', '',
              '- 28条申万社区样本已排除出默认训练输入：大部分只到2024年，采掘只到2021年，存在缺日。',
              '- 14条主要序列在各自起止范围内相对于已存参考日历无缺日，但不证明已取得该指数全部历史。中证医疗当前只从2015-08-04开始。',
              '- 中证A500和其他指数的发布前回溯及方法变更尚未审计，不等于当时已存在可交易基金。',
              f"- 沪深300东方财富与腾讯{cross['compared_closes']}条收盘中有{cross['differences_over_half_cent']}条约相差{cross['max_absolute_difference_points']}点，未强行改成一致。来源是否独立也未确认。",
              '- OHLCV统一表、估值、宏观发布时间、全收益指数和基金净值尚未补齐。当前预测目标只能明确为价格指数收益，不能直接当基金实际回报。', '',
              '## 代码入口', '',
              '- [开源参考代码及许可](../../code/references/README.md)',
              '- [项目数据准备和防泄漏切分](../../code/research/README.md)',
              '- 机器可读结论与样本数见同名JSON；联网最新观测记录见 `latest_probe_' + report['as_of'] + '.json`。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', default=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat())
    args = parser.parse_args()
    dt.date.fromisoformat(args.as_of)
    report = review(args.as_of)
    target = ROOT / 'data/reports' / f"readiness_{args.as_of}"
    market_data.write_json(target.with_suffix('.json'), report)
    target.with_suffix('.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
