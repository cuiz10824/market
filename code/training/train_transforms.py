"""Paired target-transform ablation with frozen temporal folds and original-unit scores."""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'html')]
from research.sequence_data import SequenceData, CHANNELS
from training.neural_engine import scaled_score
from training.transform_engine import fit, predict, export, load
from training.train_enhanced import read, write, sha, metrics


def csvwrite(path, records):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def group_scores(data, rows, mask, values):
    result = {}
    for ci, code in enumerate(data.codes):
        for j, h in enumerate(data.horizons):
            good = (data.ids[rows] == ci) & mask[:, j]
            if not good.any():
                raise ValueError('Empty index/horizon score')
            result[f'{code}:{h}'] = metrics(data.y[rows[good], j], values[good, j])
    return result


def historical_mean(data, train, mask, rows):
    result = np.zeros((len(rows), len(data.horizons)))
    for ci in range(len(data.codes)):
        for j in range(len(data.horizons)):
            good = (data.ids[train] == ci) & mask[:, j]
            result[data.ids[rows] == ci, j] = np.mean(data.y[train[good], j])
    return result


def run(resume=False):
    protocol = ROOT/'code/training/transform_protocol.json'
    p = read(protocol)
    torch.set_num_threads(p['threads'])
    torch.set_num_interop_threads(1)
    sources = ['code/training/train_transforms.py', 'code/training/target_transforms.py',
               'code/training/transform_engine.py', 'code/training/neural_engine.py',
               'code/research/neural_models.py', 'code/research/sequence_data.py',
               'code/training/train_enhanced.py', 'data/active_indices.json']
    source_hashes = {name: sha(ROOT/name) for name in sources}
    fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    data_sha = sha(ROOT/'data/enhanced/market_ohlcv.sqlite3')
    date = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d')
    run_id = f'transforms_{date}_{sha(protocol)[:8]}_{fingerprint[:8]}'
    models = ROOT/'code/models'/run_id
    report = ROOT/'data/reports/training'/run_id
    if (models.exists() or report.exists()) and not resume:
        raise ValueError('Run exists; use --resume to continue matching completed fits')
    provenance = dict(run_id=run_id, protocol_sha256=sha(protocol), source_sha256=source_hashes,
                      database_sha256=data_sha, research_only=True, auto_deploy=False,
                      versions={n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'pandas', 'safetensors')})
    if (report/'provenance.json').exists():
        if read(report/'provenance.json') != provenance:
            raise ValueError('Resume provenance differs')
        locked = read(report/'original_artifacts.json')
    else:
        locked = {str(f.relative_to(ROOT)): sha(f) for folder in ('code/models', 'data/reports/training')
                  for f in (ROOT/folder).rglob('*') if f.is_file()}
        for name in ('data/market.sqlite3', 'data/enhanced/market_ohlcv.sqlite3',
                     'html/daily/model_selection.json', 'html/index.html', 'html/snapshot.js'):
            locked[name] = sha(ROOT/name)
        write(report/'original_artifacts.json', locked)
        write(report/'frozen_protocol.json', p)
        write(report/'provenance.json', provenance)
    data = SequenceData(p)
    length = p['lookback']
    entries = []

    def train_save(candidate, seed, role, start, end=None, epochs=None):
        tag = f'{role}/{candidate}/{start or "all"}/{seed}'
        folder = models/tag
        result_file = folder/'completed.json'
        spec = p['candidates'][candidate]
        train, tm = data.split(start, train=True, stride=p['train_stride'])
        if len(train) < 500:
            raise ValueError('Too few training origins')
        if start and not (data.calendar[data.label_indices[train][tm]] < start).all():
            raise ValueError('Training labels cross cutoff')
        if role == 'validation':
            rows, mask = data.split(start, end)
            validation = data.split(start, end, stride=p['early_stop_validation_stride'])
        elif role == 'holdout':
            rows, mask = data.split(start)
            codes = ['000510'] if start == p['a500_holdout_start'] else [c for c in data.codes if c != '000510']
            subset = np.isin(data.ids[rows], [data.codes.index(c) for c in codes])
            rows, mask = rows[subset], mask[subset]
            validation = None
        else:
            rows = np.asarray([np.where((data.ids == ci) & (data.ends == len(data.calendar)-1))[0][0]
                               for ci in range(len(data.codes))])
            mask = data.known[rows]
            validation = None
        if result_file.exists():
            entry = read(result_file)
            for name, digest in entry['files'].items():
                if sha(ROOT/name) != digest:
                    raise ValueError('Cached artifact checksum differs')
            net, bundle = load(ROOT/entry['bundle'])
            if bundle['target_transform'] != spec:
                raise ValueError('Cached transform differs')
            with np.load(ROOT/entry['predictions'], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['rows'], rows)
                np.testing.assert_array_equal(saved['mask'], mask)
                values = saved['prediction']
            np.testing.assert_allclose(predict(net, data, length, rows, spec), values, rtol=1e-8, atol=1e-10)
            entries.append(entry)
            print('RESUMED '+tag, flush=True)
            return entry, rows, mask, values
        began = time.monotonic()
        net, log = fit(data, length, seed, train, tm, p, spec, validation, epochs)
        metadata = dict(provenance, role=role, seed=seed, cutoff=start, training_rows=len(train),
                        fit_origin_start=str(min(data.dates[train])), fit_origin_end=str(max(data.dates[train])),
                        fit_label_ends={str(h): str(max(data.calendar[data.label_indices[train, j][tm[:, j]]]))
                                        for j, h in enumerate(data.horizons)},
                        label_counts={str(h): int(tm[:, j].sum()) for j, h in enumerate(data.horizons)},
                        selected_epoch=log['selected_epoch'], epochs_executed=log['epochs_executed'])
        bundle = export(net, folder, dict(architecture=p['architecture'], lookback=length,
                        sequence_length=data.x[length].shape[1], channels=list(CHANNELS),
                        codes=data.codes, horizons=data.horizons, target_scales=data.target_scales.tolist(),
                        target_transform=spec, metadata=metadata))
        values = predict(net, data, length, rows, spec)
        reloaded, _ = load(bundle)
        np.testing.assert_allclose(predict(reloaded, data, length, rows, spec), values, rtol=1e-8, atol=1e-10)
        artifact = folder/'predictions.npz'
        np.savez_compressed(artifact, rows=rows, mask=mask, prediction=values)
        log_path = folder/'fit_log.json'
        write(log_path, log)
        entry = dict(role=role, candidate=candidate, seed=seed, start=start, end=end,
                     bundle=str(bundle.relative_to(ROOT)), predictions=str(artifact.relative_to(ROOT)),
                     log=str(log_path.relative_to(ROOT)), selected_epoch=log['selected_epoch'],
                     epochs_executed=log['epochs_executed'], elapsed_seconds=time.monotonic()-began,
                     files={str(f.relative_to(ROOT)): sha(f) for f in (bundle, folder/'weights.safetensors', artifact, log_path)})
        write(result_file, entry)
        entries.append(entry)
        print(f'{tag}: epochs={log["epochs_executed"]}, selected={log["selected_epoch"]}, seconds={entry["elapsed_seconds"]:.1f}', flush=True)
        return entry, rows, mask, values

    development = {}
    refit_epochs = {}
    # All validation fits complete before any held-out model is fitted or scored.
    for candidate in p['candidates']:
        folds = []
        selected_epochs = []
        for start, end in p['development_windows']:
            members = []
            for seed in p['seeds']:
                entry, rows, mask, values = train_save(candidate, seed, 'validation', start, end)
                selected_epochs.append(entry['selected_epoch'])
                members.append(values)
            ensemble = np.mean(members, axis=0)
            train, tm = data.split(start, train=True, stride=p['train_stride'])
            folds.append(dict(start=start, end=end, ensemble=group_scores(data, rows, mask, ensemble),
                              zero=group_scores(data, rows, mask, np.zeros_like(ensemble)),
                              mean=group_scores(data, rows, mask, historical_mean(data, train, tm, rows)),
                              scaled_score=scaled_score(data, rows, mask, ensemble)))
        development[candidate] = folds
        refit_epochs[candidate] = int(np.floor(np.median(selected_epochs)+.5))
        write(report/'development_progress.json', development)
    tasks = []
    nonlinear = [c for c in p['candidates'] if c != 'raw']
    for code in data.codes:
        for h in data.horizons:
            key = f'{code}:{h}'
            means = {c: float(np.mean([f['ensemble'][key]['mae_pp'] for f in folds]))
                     for c, folds in development.items()}
            tasks.append(dict(code=code, horizon=h, choice=min(means, key=means.get),
                              nonlinear_choice=min(nonlinear, key=means.get), validation_mae_pp=means))
    choices = dict(tasks=tasks, refit_epochs=refit_epochs, development=development,
                   policy='Validation ONLY, frozen before holdout fitting or scoring', auto_deploy=False)
    if (report/'choices.json').exists() and read(report/'choices.json') != choices:
        raise ValueError('Frozen choices changed')
    write(report/'choices.json', choices)
    print('CHOICES FROZEN '+json.dumps(refit_epochs), flush=True)
    records = []
    latest = []
    for candidate in p['candidates']:
        for start in p['holdout_starts']:
            members = []
            for seed in p['seeds']:
                _, rows, mask, values = train_save(candidate, seed, 'holdout', start, epochs=refit_epochs[candidate])
                members.append(values)
            ensemble = np.mean(members, axis=0)
            train, tm = data.split(start, train=True, stride=p['train_stride'])
            mean = historical_mean(data, train, tm, rows)
            for k, ix in enumerate(rows):
                for j, h in enumerate(data.horizons):
                    if not mask[k, j]:
                        continue
                    records.append(dict(code=data.codes[data.ids[ix]], horizon=h, date=str(data.dates[ix]),
                                        label_end_date=str(data.calendar[data.label_indices[ix, j]]),
                                        candidate=candidate, actual_return=float(data.y[ix, j]),
                                        expected_return=float(ensemble[k, j]), seed1_return=float(members[0][k, j]),
                                        seed2_return=float(members[1][k, j]), mean_return=float(mean[k, j]), zero_return=0.))
        members = []
        for seed in p['seeds']:
            _, rows, mask, values = train_save(candidate, seed, 'latest', None, epochs=refit_epochs[candidate])
            members.append(values)
        ensemble = np.mean(members, axis=0)
        for k, ix in enumerate(rows):
            for j, h in enumerate(data.horizons):
                latest.append(dict(code=data.codes[data.ids[ix]], horizon=h, as_of=str(data.dates[ix]),
                                   candidate=candidate, forecast_percent=float(ensemble[k, j]*100)))
    csvwrite(report/'test_predictions.csv', records)
    csvwrite(report/'latest_predictions.csv', latest)
    for name, digest in locked.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Original artifact changed: '+name)
    write(models/'index.json', dict(provenance, models=entries, tasks=tasks))
    write(report/'summary.json', dict(provenance, protocol=p, tasks=tasks, development=development,
                                     refit_epochs=refit_epochs, origin_count=len(data.ids), latest_as_of=str(data.calendar[-1])))
    write(report/'verification.json', dict(status='export_reload_parity_passed', native_bundles=len(entries),
          candidate_predictions=len(records), original_artifacts_unchanged=True,
          all_saved_file_sha256={str(f.relative_to(ROOT)): sha(f) for f in models.rglob('*') if f.is_file()}))
    write(ROOT/'code/models/latest_transforms.json', dict(run_id=run_id, research_only=True, auto_deploy=False,
          index=str((models/'index.json').relative_to(ROOT)), report=str((report/'summary.json').relative_to(ROOT)),
          verification=str((report/'verification.json').relative_to(ROOT))))
    print('RUN_ID='+run_id, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args().resume)
