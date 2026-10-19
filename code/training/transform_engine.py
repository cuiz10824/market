"""Isolated target-transform experiment; never changes the production neural engine."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from research.neural_models import SharedForecaster
from training.neural_engine import seed_everything, masked_loss, scaled_score
from training.target_transforms import transform, inverse


def predict(net, data, lookback, rows, spec, batch_size=256):
    net.eval()
    values = []
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            ix = rows[offset:offset+batch_size]
            values.append(net(torch.from_numpy(data.x[lookback][ix]),
                              torch.from_numpy(data.ids[ix])).cpu().numpy())
    if not values:
        return np.empty((0, len(data.horizons)))
    return inverse(np.concatenate(values), spec) * data.target_scales.astype(np.float64)


def fit(data, lookback, seed, train, train_mask, p, spec, validation=None, epochs=None):
    seed_everything(seed)
    net = SharedForecaster(p['architecture'], data.x[lookback].shape[1],
                           data.x[lookback].shape[2], len(data.codes), len(data.horizons))
    optimizer = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    rng = np.random.default_rng(seed)
    # Transform only known TRAINING targets. No fitted scale or validation labels.
    target = np.zeros(train_mask.shape, dtype=np.float64)
    normalized = data.y[train] / data.target_scales
    target[train_mask] = transform(normalized[train_mask], spec)
    target = target.astype(np.float32)
    if not np.isfinite(target).all():
        raise ValueError('Target cannot be represented in float32')
    best = None
    best_score = float('inf')
    best_epoch = None
    history = []
    stale = 0
    limit = epochs if epochs is not None else p['max_epochs']
    for epoch in range(1, limit+1):
        net.train()
        order = rng.permutation(len(train))
        losses = []
        for offset in range(0, len(order), p['batch_size']):
            local = order[offset:offset+p['batch_size']]
            ix = train[local]
            optimizer.zero_grad(set_to_none=True)
            output = net(torch.from_numpy(data.x[lookback][ix]), torch.from_numpy(data.ids[ix]))
            loss = masked_loss(output, torch.from_numpy(target[local]),
                               torch.from_numpy(train_mask[local].astype(np.float32)))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), p['gradient_clip'])
            optimizer.step()
            losses.append(float(loss.detach()))
        log = {'epoch': epoch, 'training_transformed_mae': float(np.mean(losses))}
        if validation is not None:
            vr, vm = validation
            value = scaled_score(data, vr, vm, predict(net, data, lookback, vr, spec))
            log['validation_original_scaled_mae'] = value
            if epoch >= p['minimum_epochs']:
                if value < best_score - 1e-8:
                    best_score = value
                    best_epoch = epoch
                    best = copy.deepcopy(net.state_dict())
                    stale = 0
                else:
                    stale += 1
        history.append(log)
        if validation is not None and stale >= p['patience']:
            break
    if validation is not None:
        if best is None:
            raise ValueError('No eligible validation checkpoint')
        net.load_state_dict(best)
    else:
        best_epoch = limit
    return net, {'selected_epoch': best_epoch, 'epochs_executed': len(history), 'history': history,
                 'parameter_count': sum(v.numel() for v in net.parameters())}


def export(net, folder, spec):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    native = folder / 'weights.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, str(native))
    bundle = dict(schema_version=1, task='transformed_shared_neural_return', **spec,
                  weights={'file': native.name, 'sha256': hashlib.sha256(native.read_bytes()).hexdigest()})
    path = folder / 'model.json'
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return path


def load(path):
    path = Path(path)
    bundle = json.loads(path.read_text(encoding='utf-8'))
    if bundle.get('schema_version') != 1 or bundle.get('task') != 'transformed_shared_neural_return':
        raise ValueError('Unknown transformed model')
    name = bundle['weights']['file']
    if Path(name).name != name:
        raise ValueError('Invalid native filename')
    native = path.parent / name
    if hashlib.sha256(native.read_bytes()).hexdigest() != bundle['weights']['sha256']:
        raise ValueError('Transformed weight checksum mismatch')
    # Validate transform metadata before accepting the weights.
    inverse([0.], bundle['target_transform'])
    net = SharedForecaster(bundle['architecture'], bundle['sequence_length'], len(bundle['channels']),
                           len(bundle['codes']), len(bundle['horizons']))
    net.load_state_dict(load_file(str(native), device='cpu'), strict=True)
    net.eval()
    return net, bundle
