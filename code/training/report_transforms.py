"""Original-unit, matched-date reports; test metrics never select a transform."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'html')]
from daily.selection import load_selection
from training.train_enhanced import read, write, metrics
from training.train_transforms import csvwrite

TITLES = {1: '下一交易日', 5: '一周', 21: '一月', 63: '三月', 126: '六月'}
LABELS = {'raw': '原始目标TCN', 'asinh_selected': 'asinh（验证选强度）',
          'signed_exp_selected': '双边指数（验证选强度）', 'validation_selected': '验证所选组合',
          'nonlinear_selected': '强制使用非线性组合', 'zero': '零预测', 'mean': '历史均值', 'active': '网页现用模型'}


def selection_maps(tasks, candidates):
    mappings = {'validation_selected': {key: t['choice'] for key, t in tasks.items()},
                'nonlinear_selected': {key: t['nonlinear_choice'] for key, t in tasks.items()}}
    for family in ('asinh', 'signed_exp'):
        members = [c for c in candidates if c.startswith(family)]
        mappings[family+'_selected'] = {key: min(members, key=lambda c: t['validation_mae_pp'][c])
                                       for key, t in tasks.items()}
    return mappings


def active_predictions():
    manifest = read(ROOT/'html/daily/model_selection.json')
    pieces = []
    for source, entry in manifest['sources'].items():
        path = (ROOT/entry['report']).parent/'test_predictions.csv'
        frame = pd.read_csv(path, dtype={'code': str})
        for task in manifest['tasks']:
            if task['source'] != source:
                continue
            selected = frame[(frame.code == task['code']) & (frame.horizon == task['horizon']) &
                             (frame.candidate == task['candidate'])]
            if selected.empty:
                raise ValueError('Missing active-model reference')
            pieces.append(selected[['code', 'horizon', 'date', 'actual_return', 'expected_return']])
    return pd.concat(pieces, ignore_index=True)


def report(pointer=None):
    pointer = read(ROOT/'code/models/latest_transforms.json') if pointer is None else pointer
    s = read(ROOT/pointer['report'])
    folder = (ROOT/pointer['report']).parent
    p = s['protocol']
    tasks = {(t['code'], t['horizon']): t for t in s['tasks']}
    names = {i['code']: i['display_name'] for i in load_selection(ROOT)}
    frame = pd.read_csv(folder/'test_predictions.csv', dtype={'code': str})
    base = frame[frame.candidate == 'raw'].copy()
    keys = ['code', 'horizon', 'date']
    if frame.duplicated(keys+['candidate']).any():
        raise ValueError('Duplicate test origin')
    original_index = pd.MultiIndex.from_frame(base[keys])
    for candidate in p['candidates']:
        part = frame[frame.candidate == candidate]
        pd.testing.assert_index_equal(pd.MultiIndex.from_frame(part[keys]), original_index)
        np.testing.assert_allclose(part.actual_return, base.actual_return, rtol=0, atol=1e-12)
    # Freeze mapping from development choices only; do not choose on held-out errors.
    mappings = selection_maps(tasks, p['candidates'])
    pieces = [frame]
    for label, mapping in mappings.items():
        chosen = [r.candidate == mapping[(r.code, r.horizon)] for r in frame.itertuples()]
        part = frame.loc[chosen].copy()
        part['candidate'] = label
        pieces.append(part)
    for label in ('zero', 'mean'):
        part = base.copy()
        part['expected_return'] = part[label+'_return']
        part['candidate'] = label
        pieces.append(part)
    reference = active_predictions()
    part = base.merge(reference, on=keys, how='left', validate='one_to_one', suffixes=('', '_active'))
    if part.expected_return_active.isna().any():
        raise ValueError('Active reference dates incomplete')
    np.testing.assert_allclose(part.actual_return, part.actual_return_active, rtol=0, atol=1e-12)
    part['expected_return'] = part.expected_return_active
    part['candidate'] = 'active'
    pieces.append(part[base.columns])
    all_predictions = pd.concat(pieces, ignore_index=True)

    def score(group):
        result = metrics(group.actual_return, group.expected_return)
        error = (group.expected_return.to_numpy()-group.actual_return.to_numpy())*100
        result['absolute_error_std_pp'] = float(np.std(np.abs(error), ddof=0))
        result['mean_absolute_forecast_pp'] = float(np.mean(np.abs(group.expected_return))*100)
        result['near_zero_fraction'] = float(np.mean(np.abs(group.expected_return)*100 < .01))
        return result

    flat = []
    yearly = []
    for (code, h, candidate), group in all_predictions.groupby(['code', 'horizon', 'candidate'], sort=False):
        raw = base[(base.code == code) & (base.horizon == h)]
        raw_mae = metrics(raw.actual_return, raw.expected_return)['mae_pp']
        zero_mae = metrics(raw.actual_return, np.zeros(len(raw)))['mae_pp']
        scored = score(group)
        flat.append(dict(code=code, name=names[code], horizon=int(h), candidate=candidate, **scored,
                         raw_mae_pp=raw_mae, zero_mae_pp=zero_mae,
                         mae_gain_vs_raw_pp=raw_mae-scored['mae_pp'], mae_gain_vs_zero_pp=zero_mae-scored['mae_pp'],
                         validation_choice=tasks[(code, h)]['choice']))
        for year, subset in group.groupby(group.date.str[:4]):
            yearly.append(dict(code=code, name=names[code], horizon=int(h), candidate=candidate,
                               origin_year=year, **score(subset)))
    csvwrite(folder/'candidate_metrics.csv', flat)
    csvwrite(folder/'yearly_metrics.csv', yearly)
    table = pd.DataFrame(flat)
    aggregates = []
    for (h, candidate), group in table.groupby(['horizon', 'candidate'], sort=False):
        fields = ['mae_pp', 'bias_pp', 'error_std_pp', 'absolute_error_std_pp', 'rmse_pp',
                  'mean_absolute_forecast_pp', 'near_zero_fraction']
        aggregates.append(dict(horizon=int(h), candidate=candidate, **{f: float(group[f].mean()) for f in fields},
                               index_count=len(group), beat_zero_tasks=int((group.mae_pp < group.zero_mae_pp).sum()),
                               beat_raw_tasks=int((group.mae_pp < group.raw_mae_pp-1e-12).sum())))
    csvwrite(folder/'aggregate_metrics.csv', aggregates)
    seed_scores = []
    for (code, h, candidate), group in frame.groupby(['code', 'horizon', 'candidate'], sort=False):
        for member in ('seed1_return', 'seed2_return', 'expected_return'):
            seed_scores.append(dict(code=code, name=names[code], horizon=int(h), candidate=candidate, member=member,
                                    **metrics(group.actual_return, group[member])))
    csvwrite(folder/'seed_metrics.csv', seed_scores)
    latest = pd.read_csv(folder/'latest_predictions.csv', dtype={'code': str})
    selected_latest = []
    for row in latest.itertuples():
        if row.candidate == tasks[(row.code, row.horizon)]['choice']:
            selected_latest.append(dict(code=row.code, name=names[row.code], horizon=row.horizon, as_of=row.as_of,
                                        candidate=row.candidate, forecast_percent=row.forecast_percent))
    csvwrite(folder/'validation_selected_latest.csv', selected_latest)
    scaled_mae = {c: float(np.mean([r['mae_pp']/np.sqrt(r['horizon']) for r in aggregates if r['candidate'] == c]))
                  for c in table.candidate.unique()}
    comparison = dict(run_id=s['run_id'], aggregate_by_horizon=aggregates, test_equal_index_horizon_scaled_mae=scaled_mae,
                      candidate_task_counts={c: sum(t['choice'] == c for t in s['tasks']) for c in p['candidates']},
                      auto_deploy=False, test_status=p['test_status'])
    write(folder/'comparison.json', comparison)
    validation_rows = []
    for candidate, folds in s['development'].items():
        for fold in folds:
            for key, m in fold['ensemble'].items():
                code, h = key.split(':')
                validation_rows.append(dict(code=code, name=names[code], horizon=int(h), candidate=candidate,
                    fold_start=fold['start'], fold_end=fold['end'], **m,
                    zero_mae_pp=fold['zero'][key]['mae_pp'], mean_mae_pp=fold['mean'][key]['mae_pp']))
    csvwrite(folder/'validation_metrics.csv', validation_rows)
    lookup = {(r['horizon'], r['candidate']): r for r in aggregates}
    main = ['zero', 'raw', 'asinh_selected', 'signed_exp_selected', 'validation_selected', 'active']
    asinh_stats = table[table.candidate == 'asinh_selected']
    chosen_stats = table[table.candidate == 'validation_selected']
    asinh_wins_raw = int((asinh_stats.mae_pp < asinh_stats.raw_mae_pp-1e-12).sum())
    asinh_wins_zero = int((asinh_stats.mae_pp < asinh_stats.zero_mae_pp).sum())
    chosen_wins_zero = int((chosen_stats.mae_pp < chosen_stats.zero_mae_pp).sum())
    lines = ['# 非线性训练目标实验', '', f'运行 `{s["run_id"]}`；网页模型未更换。', '',
             '## 结果判断', '',
             f'- asinh在{asinh_wins_raw}/40项任务上改善原始TCN的MAE，但仅{asinh_wins_zero}/40项优于零预测，多数改善幅度较小。',
             f'- 验证所选组合仅{chosen_wins_zero}/40项优于零预测；不能把验证中的胜出直接当成测试中的优势。',
             f'- 半年验证所选组合MAE为{lookup[(126,"validation_selected")]["mae_pp"]:.4f}个百分点，网页现用模型对应回测为{lookup[(126,"active")]["mae_pp"]:.4f}，当前模型更低。',
             '- 这轮历史对照没有显示仅靠目标变换就能全面解决预测偏弱的问题，未自动替换网页。零预测较强也不等于市场波动小：三个月零预测MAE仍为'+f'{lookup[(63,"zero")]["mae_pp"]:.4f}个百分点。', '',
             '## 实验与时间隔离', '',
             '- 固定小TCN、756交易日输入（约三年）、8指数共享网络、五个累计收益输出头；与前轮神经网络实验使用相同满足1260+1完整历史的起点。只改变训练目标，架构、输入、采样、初始化种子、优化器和训练预算一致。',
             '- 标签为r=close[t+h]/close[t]-1，先按固定s=0.01×sqrt(h)归一化为u。原始目标为u；asinh为asinh(αu)/α，α∈{0.5,1}；双边指数为sign(u)×expm1(α|u|)/α，α∈{0.1,0.25}。五种映射在零处斜率都为1，以区分曲率效果和简单改梯度尺度；asinh仍会提高近零差异相对于大涨跌的权重。',
             '- 训练损失为变换空间的逐头等权MAE。所有早停、强度选择和最终性能都先逆变换，使用原始涨跌幅；不把变换空间损失与收益百分点混比，不放大最终输出，不剪裁异常收益。',
             '- 三个滚动验证窗：2018—2019、2020—2021、2022—2023；每头训练标签结束日严格早于切分点。参数网格固定，不根据测试成绩修改。两个固定种子分别训练，先逆变换再对收益输出等权平均。',
             '- 共30个验证模型。每个候选根据六次验证所选epoch的中位数确定重训轮数；在任何测试拟合或评分前冻结40项选择和重训预算。另训练20个测试模型、10个截至最新数据的研究模型，总共60个。',
             '- 测试起点2024年，A500起点2025年；只评分未来标签已经完整落地的样本。此前已反复观察这些年份，因此本次是历史探索性对照，不能称为全新独立测试。期限重叠、行业相关，未做显著性结论。',
             '- asinh/双边指数两列分别用验证选取各自强度；“验证所选组合”允许原始目标获胜，逐行业逐期限选择。全部测试候选、强制非线性组合、历史均值和双种子结果均保留。网页现用模型只作同日期参考，不参与本轮选择。', '',
             '“网页现用模型”使用其对应研究实验在测试切分点重训后保存的历史预测，匹配同一行业、期限、预测日期和真实收益；没有用当前全量训练权重回推历史。', '',
             '## 平均绝对误差MAE（百分点，8指数等权）', '',
             '|期限|'+'|'.join(LABELS[k] for k in main)+'|', '|---|'+'---:|'*len(main)]
    for h in p['horizons']:
        lines.append('|'+TITLES[h]+'|'+'|'.join(f'{lookup[(h,k)]["mae_pp"]:.4f}' for k in main)+'|')
    lines += ['', '## 带符号预测误差的标准差（百分点，8指数等权）', '',
              '|期限|'+'|'.join(LABELS[k] for k in main)+'|', '|---|'+'---:|'*len(main)]
    for h in p['horizons']:
        lines.append('|'+TITLES[h]+'|'+'|'.join(f'{lookup[(h,k)]["error_std_pp"]:.4f}' for k in main)+'|')
    lines += ['', '标准差为ddof=0，先逐指数计算再平均。MAE是绝对误差均值，error_std_pp是带符号误差标准差；absolute_error_std_pp另给出绝对误差标准差。偏差均值、RMSE、预测绝对幅度和接近零比例同时保留，不能只看数字是否变大。', '',
              '## 验证所选组合：分行业、分期限MAE', '',
              '|行业|'+'|'.join(TITLES[h] for h in p['horizons'])+'|', '|---|'+'---:|'*len(p['horizons'])]
    for code in p['codes']:
        values = []
        for h in p['horizons']:
            row = table[(table.code == code) & (table.horizon == h) & (table.candidate == 'validation_selected')].iloc[0]
            values.append(f'{row.mae_pp:.4f}')
        lines.append('|'+names[code]+'|'+'|'.join(values)+'|')
    lines += ['', '## 选择与复验', '', '验证选择数量：'+', '.join(f'{c}={n}' for c,n in comparison['candidate_task_counts'].items())+'。', '',
              '|方案|40项中测试MAE优于零|40项中优于原始TCN|', '|---|---:|---:|']
    for candidate in main[1:]:
        selected = table[table.candidate == candidate]
        lines.append(f'|{LABELS[candidate]}|{int((selected.mae_pp < selected.zero_mae_pp).sum())}/40|{int((selected.mae_pp < selected.raw_mae_pp-1e-12).sum())}/40|')
    lines += ['', '单调目标变换不会创造输入信息。对无限容量、充分优化的模型，严格单调变换下MAE对应的条件中位数逆变换后仍相同；这里的差异来自有限网络、损失权重和优化路径。幅度变大本身不是准确率改善的证据。', '',
              '复验：`python code/training/verify_transforms.py`；重新生成报告：`python code/training/report_transforms.py`。', '',
              '明细：candidate_metrics.csv（全部方案×行业×期限）、yearly_metrics.csv（按预测起点年份）、seed_metrics.csv（双种子）、test_predictions.csv（每个真实值和预测值）、validation_selected_latest.csv（研究用最新输出）、fit日志和safetensors权重。']
    (folder/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[lines.index('## 平均绝对误差MAE（百分点，8指数等权）'):lines.index('## 带符号预测误差的标准差（百分点，8指数等权）')]))
    print('REPORT='+str(folder.relative_to(ROOT)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run')
    args = parser.parse_args()
    report({'report': f'data/reports/training/{args.run}/summary.json'} if args.run else None)
