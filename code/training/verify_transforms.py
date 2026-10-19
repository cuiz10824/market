"""Independently reload every native checkpoint and reproduce the reported predictions."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'html')]
from research.sequence_data import SequenceData
from training.transform_engine import load, predict
from training.train_enhanced import read, write, sha, metrics


def verify(pointer=None):
    pointer = read(ROOT/'code/models/latest_transforms.json') if pointer is None else pointer
    summary = read(ROOT/pointer['report'])
    folder = (ROOT/pointer['report']).parent
    p = summary['protocol']
    torch.set_num_threads(p['threads'])
    torch.set_num_interop_threads(1)
    for source, expected in summary['source_sha256'].items():
        if sha(ROOT/source) != expected:
            raise ValueError('Source fingerprint differs: '+source)
    if sha(ROOT/'data/enhanced/market_ohlcv.sqlite3') != summary['database_sha256']:
        raise ValueError('Dataset fingerprint differs')
    if sha(folder/'frozen_protocol.json') != summary['protocol_sha256']:
        raise ValueError('Protocol differs')
    for name, digest in read(folder/'original_artifacts.json').items():
        if sha(ROOT/name) != digest:
            raise ValueError('Original artifact changed: '+name)
    check = read(ROOT/pointer['verification'])
    for name, digest in check['all_saved_file_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Saved artifact differs: '+name)
    data = SequenceData(p)
    index = read(ROOT/pointer['index'])
    cache = {}
    replayed = 0
    original_pointer = read(ROOT/'code/models/latest_neural.json')
    original_index = read(ROOT/original_pointer['index'])
    original_raw = {}
    for entry in original_index['models']:
        if entry['architecture'] != p['architecture'] or entry['lookback'] != p['lookback']:
            continue
        role = 'validation' if entry['role'] in ('screen', 'replication') else entry['role']
        original_raw[(role, entry['start'], entry['seed'])] = entry
    raw_parity = 0
    for entry in index['models']:
        net, bundle = load(ROOT/entry['bundle'])
        if bundle['target_transform'] != p['candidates'][entry['candidate']]:
            raise ValueError('Transform metadata differs')
        start = entry['start']
        train, mask = data.split(start, train=True, stride=p['train_stride'])
        m = bundle['metadata']
        if m['training_rows'] != len(train):
            raise ValueError('Training coverage differs')
        for j, h in enumerate(data.horizons):
            label_end = str(max(data.calendar[data.label_indices[train, j][mask[:, j]]]))
            if m['fit_label_ends'][str(h)] != label_end or (start and label_end >= start):
                raise ValueError('Training cutoff violation')
        with np.load(ROOT/entry['predictions'], allow_pickle=False) as saved:
            rows = saved['rows']
            valid = saved['mask']
            expected = saved['prediction']
        # Recreate split and horizon masks rather than trusting saved arrays.
        if entry['role'] == 'validation':
            wanted, wanted_mask = data.split(start, entry['end'])
        elif entry['role'] == 'holdout':
            wanted, wanted_mask = data.split(start)
            codes = ['000510'] if start == p['a500_holdout_start'] else [c for c in data.codes if c != '000510']
            subset = np.isin(data.ids[wanted], [data.codes.index(c) for c in codes])
            wanted, wanted_mask = wanted[subset], wanted_mask[subset]
        else:
            wanted = np.asarray([np.where((data.ids == ci) & (data.ends == len(data.calendar)-1))[0][0]
                                 for ci in range(len(data.codes))])
            wanted_mask = data.known[wanted]
        np.testing.assert_array_equal(rows, wanted)
        np.testing.assert_array_equal(valid, wanted_mask)
        values = predict(net, data, p['lookback'], rows, bundle['target_transform'])
        np.testing.assert_allclose(values, expected, rtol=1e-8, atol=1e-10)
        replayed += int(valid.sum())
        key = (entry['role'], entry['candidate'], start)
        cache.setdefault(key, []).append((rows, valid, values, entry['seed']))
        if entry['candidate'] == 'raw':
            previous = original_raw[(entry['role'], start, entry['seed'])]
            artifact = previous['validation' if entry['role'] == 'validation' else 'test' if entry['role'] == 'holdout' else 'latest']
            with np.load(ROOT/artifact, allow_pickle=False) as saved:
                np.testing.assert_array_equal(rows, saved['rows'])
                np.testing.assert_allclose(values, saved['prediction'], rtol=1e-8, atol=1e-10)
            raw_parity += 1
        print(f'replayed {entry["role"]}/{entry["candidate"]}/{start or "all"}/{entry["seed"]}', flush=True)
    observed = pd.read_csv(folder/'test_predictions.csv', dtype={'code': str})
    latest = pd.read_csv(folder/'latest_predictions.csv', dtype={'code': str})
    latest.set_index(['candidate', 'code', 'horizon'], inplace=True)
    csv_replayed = 0
    for (role, candidate, start), members in cache.items():
        if len(members) != len(p['seeds']):
            raise ValueError('Missing ensemble member')
        members.sort(key=lambda m: p['seeds'].index(m[3]))
        rows, valid = members[0][:2]
        for other in members[1:]:
            np.testing.assert_array_equal(rows, other[0])
            np.testing.assert_array_equal(valid, other[1])
        values = np.mean([m[2] for m in members], axis=0)
        if role == 'latest':
            for k, ix in enumerate(rows):
                for j, h in enumerate(data.horizons):
                    saved = latest.loc[(candidate, data.codes[data.ids[ix]], h)]
                    np.testing.assert_allclose(saved.forecast_percent, values[k,j]*100, rtol=1e-10, atol=1e-10)
            continue
        for ci, code in enumerate(data.codes):
            for j, h in enumerate(data.horizons):
                good = (data.ids[rows] == ci) & valid[:, j]
                if not good.any():
                    continue
                if role == 'validation':
                    fold = next(f for f in summary['development'][candidate] if f['start'] == start)
                    expected = fold['ensemble'][f'{code}:{h}']
                    actual = metrics(data.y[rows[good], j], values[good, j])
                    for field in actual:
                        np.testing.assert_allclose(actual[field], expected[field], rtol=1e-9, atol=1e-9)
                else:
                    part = observed[(observed.code == code) & (observed.horizon == h) & (observed.candidate == candidate)]
                    np.testing.assert_array_equal(part.date.to_numpy(), data.dates[rows[good]])
                    np.testing.assert_array_equal(part.label_end_date.to_numpy(), data.calendar[data.label_indices[rows[good], j]])
                    np.testing.assert_allclose(part.actual_return, data.y[rows[good], j], rtol=1e-10, atol=1e-12)
                    np.testing.assert_allclose(part.expected_return, values[good, j], rtol=1e-9, atol=1e-11)
                    # Recompute truth directly from frozen OHLCV closes.
                    raw = data.raw[code]
                    truth = raw[data.label_indices[rows[good], j], 3]/raw[data.ends[rows[good]], 3]-1
                    np.testing.assert_allclose(part.actual_return, truth, rtol=1e-10, atol=1e-12)
                    csv_replayed += len(part)
    if csv_replayed != len(observed) or len(index['models']) != 60 or raw_parity != 12:
        raise ValueError('Replay coverage incomplete')
    choices = read(folder/'choices.json')
    for task in choices['tasks']:
        means = {c: float(np.mean([f['ensemble'][f'{task["code"]}:{task["horizon"]}']['mae_pp']
                                  for f in summary['development'][c]])) for c in p['candidates']}
        if task['choice'] != min(means, key=means.get):
            raise ValueError('Choice differs from validation minimum')
        for candidate in p['candidates']:
            np.testing.assert_allclose(means[candidate], task['validation_mae_pp'][candidate], atol=1e-12)
    report_metrics_verified = 0
    if (folder/'candidate_metrics.csv').exists():
        reported = pd.read_csv(folder/'candidate_metrics.csv', dtype={'code': str})
        for row in reported.itertuples():
            if row.candidate not in p['candidates']:
                continue
            group = observed[(observed.code == row.code) & (observed.horizon == row.horizon) &
                             (observed.candidate == row.candidate)]
            expected = metrics(group.actual_return, group.expected_return)
            for field, value in expected.items():
                np.testing.assert_allclose(getattr(row, field), value, rtol=1e-10, atol=1e-10)
            error = (group.expected_return.to_numpy()-group.actual_return.to_numpy())*100
            np.testing.assert_allclose(row.absolute_error_std_pp, np.std(np.abs(error), ddof=0), atol=1e-10)
            report_metrics_verified += 1
        aggregate = pd.read_csv(folder/'aggregate_metrics.csv')
        for row in aggregate.itertuples():
            group = reported[(reported.horizon == row.horizon) & (reported.candidate == row.candidate)]
            if len(group) != len(p['codes']):
                raise ValueError('Aggregate does not equally cover8 indices')
            for field in ('mae_pp', 'bias_pp', 'error_std_pp', 'absolute_error_std_pp', 'rmse_pp'):
                np.testing.assert_allclose(getattr(row, field), group[field].mean(), atol=1e-10)
        if report_metrics_verified != 200:
            raise ValueError('Candidate report coverage incomplete')
    result = dict(status='independent_replay_passed', native_bundles=60, raw_prior_experiment_parity=raw_parity,
                  replayed_masked_pairs=replayed, test_csv_pairs=csv_replayed,
                  candidate_metric_rows_verified=report_metrics_verified,
                  training_cutoffs_verified=True, choices_validation_only=True, original_artifacts_unchanged=True,
                  production_models_unchanged=True)
    write(folder/'independent_verification.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run')
    args = parser.parse_args()
    if args.run:
        folder = f'data/reports/training/{args.run}'
        verify(dict(report=folder+'/summary.json', index=f'code/models/{args.run}/index.json', verification=folder+'/verification.json'))
    else:
        verify()
