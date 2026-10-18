"""Offline training/evaluation; freezes model selection before final test scores."""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from inference.predictor import Predictor
from research.dataset import FEATURES, build_feature_rows, build_samples, load_history
from training.engine import (choose, direction_metrics, fit, predict, return_metrics,
                             save_bundle, split_window)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def development(rows, protocol):
    scores = {'direction': {k: [] for k in protocol['classification_candidates']},
              'return': {k: [] for k in protocol['regression_candidates']}}
    folds = []
    for start, end in protocol['development_windows']:
        split = split_window(rows, start, end, protocol['min_train_samples'], protocol['min_validation_samples'])
        if split is None:
            folds.append({'start': start, 'end': end, 'status': 'skipped_insufficient_history'})
            continue
        train, valid = split
        fold = {'start': start, 'end': end, 'status': 'completed', 'train_n': len(train),
                'validation_n': len(valid), 'last_train_label_end': train[-1]['label_end_date'],
                'last_validation_label_end': valid[-1]['label_end_date'], 'scores': {}}
        for target in scores:
            fold['scores'][target] = {}
            for kind in scores[target]:
                fitted = fit(kind, target, train, protocol)
                values = predict(fitted, target, valid)
                metric = (direction_metrics(valid, values, valid[0]['horizon'])['brier'] if target == 'direction'
                          else return_metrics(valid, values)['mae'])
                scores[target][kind].append(metric)
                fold['scores'][target][kind] = metric
        folds.append(fold)
    if not scores['direction']['constant']:
        raise ValueError('No completed development folds')
    choices, averages = {}, {}
    for target in scores:
        choices[target], averages[target] = choose(scores[target], target, protocol)
    return choices, averages, folds


def parity(bundle, rows, direction, returns):
    output = Predictor(bundle).predict(rows)
    for target, key, fitted in [('direction', 'p_up', direction), ('return', 'expected_price_return', returns)]:
        if not np.allclose([p[key] for p in output], predict(fitted, target, rows), rtol=1e-10, atol=1e-12):
            raise ValueError('Reloaded model differs from fitted estimator: ' + target)


def train_task(code, horizon, rows, live_feature, meta, protocol, model_root, provenance):
    choices, averages, folds = development(rows, protocol)
    # No final-test data are passed into development or choose().
    holdout_start = protocol['holdout_overrides'].get(code, protocol['holdout_start'])
    if any(end > holdout_start for _, end in protocol['development_windows']):
        raise ValueError('Development extends into final holdout')
    split = split_window(rows, holdout_start, min_train=protocol['min_train_samples'],
                         min_test=protocol['min_validation_samples'])
    if split is None:
        raise ValueError('Insufficient final holdout data')
    train, test = split
    models = {target: fit(choices[target], target, train, protocol) for target in choices}
    p = predict(models['direction'], 'direction', test)
    returns = predict(models['return'], 'return', test)
    baseline_p = predict(fit('constant', 'direction', train, protocol), 'direction', test)
    momentum_p = predict(fit('momentum', 'direction', train, protocol), 'direction', test)
    mean_r = predict(fit('mean', 'return', train, protocol), 'return', test)
    zero_r = np.zeros(len(test))
    dm = direction_metrics(test, p, horizon)
    base_dm = direction_metrics(test, baseline_p, horizon)
    rm = return_metrics(test, returns)
    zero_rm = return_metrics(test, zero_r)
    holdout_metadata = dict(provenance, role='frozen_final_test_model', research_only=True,
                            fit_samples=len(train), fit_feature_end=train[-1]['date'],
                            fit_label_end=train[-1]['label_end_date'],
                            holdout_start=holdout_start, index_name=meta['name'],
                            data_warnings=meta['quality']['warnings'])
    holdout_bundle = save_bundle(model_root / 'holdout' / code / str(horizon), code, horizon,
                                 models['direction'], models['return'], holdout_metadata)
    parity(holdout_bundle, test[:10], models['direction'], models['return'])
    # Refit ONLY the previously chosen model kinds for future use. Final-test scores
    # never alter kind, hyperparameters, thresholds or features.
    latest = {target: fit(choices[target], target, rows, protocol) for target in choices}
    latest_metadata = dict(provenance, role='latest_refit_for_future_inference', research_only=True,
                           fit_samples=len(rows), fit_feature_end=rows[-1]['date'],
                           fit_label_end=rows[-1]['label_end_date'], inference_as_of=live_feature['date'],
                           index_name=meta['name'], data_warnings=meta['quality']['warnings'],
                           note='Final-test metrics belong to holdout model, NOT this all-labeled-data refit. Probability calibration and return intervals not implemented.')
    latest_bundle = save_bundle(model_root / 'latest' / code / str(horizon), code, horizon,
                                latest['direction'], latest['return'], latest_metadata)
    parity(latest_bundle, [live_feature], latest['direction'], latest['return'])
    result = {'code': code, 'name': meta['name'], 'horizon': horizon, 'choices': choices,
              'development_mean_scores': averages, 'development_folds': folds,
              'test': {'start': test[0]['date'], 'end': test[-1]['date'],
                       'label_end': test[-1]['label_end_date'], 'train_n': len(train),
                       'selected_direction': dm, 'historical_frequency_direction': base_dm,
                       'momentum_direction': direction_metrics(test, momentum_p, horizon),
                       'always_up_accuracy': dm['up_rate'],
                       'selected_return': rm, 'zero_return': zero_rm, 'historical_mean_return': return_metrics(test, mean_r),
                       'brier_improvement_vs_frequency': base_dm['brier'] - dm['brier'],
                       'mae_improvement_vs_zero': zero_rm['mae'] - rm['mae']},
              'holdout_bundle': holdout_bundle.relative_to(ROOT).as_posix(),
              'latest_bundle': latest_bundle.relative_to(ROOT).as_posix(),
              'latest_fit_samples': len(rows), 'latest_fit_feature_end': rows[-1]['date'],
              'latest_known_label_end': rows[-1]['label_end_date'],
              'inference_as_of': live_feature['date'], 'save_reload_parity_passed': True}
    predictions = [dict(code=code, horizon=horizon, date=r['date'], label_end_date=r['label_end_date'],
                        actual_return=r['target_return'], actual_up=r['target_up'], p_up=float(pp),
                        baseline_p_up=float(bp), momentum_p_up=float(mp), expected_return=float(rr),
                        zero_return=0., historical_mean_return=float(mr))
                   for r, pp, bp, mp, rr, mr in zip(test, p, baseline_p, momentum_p, returns, mean_r)]
    return result, predictions


def report_markdown(report):
    lines = ['# 离线训练与最终测试报告', '', f"运行：`{report['run_id']}`；数据截至 {report['data_end']}。", '',
             '先冻结协议和超参数，按2018—2019、2020—2021、2022—2023滚动验证选模型；历史不足的折明确跳过。最终测试从2024年起，中证A500从2025年起。所有训练标签在验证/测试开始前结束，验证标签也不得越过窗口末端。', '',
             '分类按Brier分数选择，回归按MAE选择；浅层LightGBM、逻辑回归、岭回归没有优势时保留简单基准。最终测试不参与选择或调参。', '',
             '| 期限（交易日） | 任务数 | Brier优于历史比例 | MAE优于零收益 | 平均Brier改善 | 平均MAE改善（百分点） |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for h, m in report['aggregate_by_horizon'].items():
        lines.append(f"| {h} | {m['tasks']} | {m['direction_better']} | {m['return_better']} | {m['mean_brier_improvement']:.5f} | {100*m['mean_mae_improvement']:.3f} |")
    lines += ['', '正改善才表示优于相应基准。这是逐指数等权描述，不是独立样本的显著性结论。', '',
              '| 指数 | 期限 | 分类模型 | 测试准确率 | 历史比例基准准确率 | Brier改善 | 回归模型 | MAE（百分点） | 非重叠样本数 |',
              '| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |']
    for r in report['tasks']:
        t = r['test']
        lines.append(f"| {r['name']} | {r['horizon']} | {r['choices']['direction']} | {t['selected_direction']['accuracy']:.1%} | {t['historical_frequency_direction']['accuracy']:.1%} | {t['brier_improvement_vs_frequency']:.5f} | {r['choices']['return']} | {100*t['selected_return']['mae']:.2f} | {t['selected_direction']['nonoverlap_n']} |")
    lines += ['', '## 如何理解模型文件', '',
              '- `holdout/`保留最终测试时的固定模型，可重放报告；`latest/`在测试完成后按已选种类重新拟合全部已知标签，供未来推理。测试成绩不能冒充latest重训模型的独立成绩。',
              '- 模型保存为JSON参数和LightGBM原生文本，附特征顺序、数据/协议/源码指纹和训练截止日期，无需反序列化pickle。',
              '- 每个保存模型均重载并核对预测一致性；末尾未知标签没有补造，半年模型的训练特征截止自然早于数据截止约半年。', '',
              '## 限制', '',
              '- 价格数据与指数回溯/方法变更尚未完成正式时点审计。A500使用回溯历史进行开发，只将2025年后的观测计入最终测试；不据此宣称历史可交易。',
              '- 半年标签高度重叠，逐日准确率不能当成大量独立实验；同时列出非重叠观测数量，不提供不可靠的显著性或收益保证。',
              '- 只有10个价格特征，没有统一宏观、估值、基金净值、分红或费用，不计算可实现交易收益。',
              '- 概率未另行校准，收益区间未实现。模型可能不优于基准，不能将单个阶段的好成绩概括成稳定预测能力。',
              '- 本轮测试结果已被观察；后续据此改动属于新实验，必须保留本轮报告，并使用新的未观察数据或明确标注反复测试偏差。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codes', nargs='+')
    args = parser.parse_args()
    protocol_path = ROOT / 'code/training/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    codes = args.codes or protocol['codes']
    if set(codes) - set(protocol['codes']) or len(codes) != len(set(codes)):
        parser.error('Unsupported or duplicate index codes')
    db_hash, protocol_hash = digest(ROOT / 'data/market.sqlite3'), digest(protocol_path)
    sources = ['code/training/train.py', 'code/training/engine.py', 'code/research/dataset.py', 'code/inference/predictor.py']
    source_hashes = {p: digest(ROOT / p) for p in sources}
    fingerprint = hashlib.sha256(json.dumps([source_hashes, codes], sort_keys=True).encode()).hexdigest()
    date = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d')
    run_id = f'{date}_{protocol_hash[:8]}_{db_hash[:8]}_{fingerprint[:8]}'
    model_root = ROOT / 'code/models' / run_id
    output = ROOT / 'data/reports/training' / run_id
    if model_root.exists() or output.exists():
        raise FileExistsError('Run already exists; preserve it. Use its saved models/report or a new protocol version.')
    output.mkdir(parents=True)
    provenance = {'protocol_sha256': protocol_hash, 'database_sha256': db_hash, 'source_sha256': source_hashes,
                  'run_id': run_id, 'python': platform.python_version(),
                  'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm']}}
    write_json(output / 'frozen_protocol.json', protocol)
    write_json(output / 'provenance.json', provenance)
    started = time.monotonic()
    tasks, all_predictions = [], []
    try:
        for code in codes:
            calendar, prices, meta = load_history(code)
            features = build_feature_rows(calendar, prices, code=code)
            cache = {r['date']: r for r in features}
            for horizon in protocol['horizons']:
                rows = build_samples(calendar, prices, horizon, code=code, feature_cache=cache)
                result, predictions = train_task(code, horizon, rows, features[-1], meta, protocol, model_root, provenance)
                tasks.append(result)
                all_predictions.extend(predictions)
                write_json(output / 'progress.json', {'completed': len(tasks), 'expected': len(codes)*len(protocol['horizons']), 'last_task': [code,horizon]})
                print(f"[{len(tasks)}/{len(codes)*len(protocol['horizons'])}] {code} h={horizon} selected={result['choices']} test_Brier_gain={result['test']['brier_improvement_vs_frequency']:.5f}", flush=True)
        if digest(protocol_path) != protocol_hash or digest(ROOT / 'data/market.sqlite3') != db_hash:
            raise ValueError('Protocol or database changed during training')
        aggregate = {}
        for horizon in protocol['horizons']:
            group = [r for r in tasks if r['horizon'] == horizon]
            aggregate[str(horizon)] = {'tasks': len(group),
                'direction_better': sum(r['test']['brier_improvement_vs_frequency'] > 1e-6 for r in group),
                'return_better': sum(r['test']['mae_improvement_vs_zero'] > 1e-6 for r in group),
                'mean_brier_improvement': float(np.mean([r['test']['brier_improvement_vs_frequency'] for r in group])),
                'mean_mae_improvement': float(np.mean([r['test']['mae_improvement_vs_zero'] for r in group]))}
        report = dict(provenance, run_id=run_id, data_end=max(r['inference_as_of'] for r in tasks),
                      status='completed_research_only', tasks=tasks, aggregate_by_horizon=aggregate,
                      elapsed_seconds=round(time.monotonic()-started,2),
                      git_parent_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip())
        write_json(output / 'summary.json', report)
        (output / 'README.md').write_text(report_markdown(report), encoding='utf-8')
        with (output / 'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(all_predictions[0]))
            writer.writeheader();writer.writerows(all_predictions)
        index = {'run_id': run_id, 'research_only': True, 'protocol_sha256': protocol_hash,
                 'report': (output/'summary.json').relative_to(ROOT).as_posix(),
                 'models': [{'code':r['code'],'horizon':r['horizon'],'latest':r['latest_bundle'],
                             'holdout':r['holdout_bundle']} for r in tasks]}
        write_json(model_root / 'index.json', index)
        print('COMPLETED '+run_id,flush=True)
    except Exception as exc:
        write_json(output/'failure.json',{'status':'failed','completed_tasks':len(tasks),'error':repr(exc)})
        raise


if __name__ == '__main__':
    main()
