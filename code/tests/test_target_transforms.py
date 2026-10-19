"""Nonlinear target inversion, original-unit scoring and isolated persistence."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from training.target_transforms import transform, inverse
from training import neural_engine, transform_engine
from training.report_transforms import selection_maps


def fixture():
    rng = np.random.default_rng(1)
    data = SimpleNamespace(x={20: rng.normal(size=(32, 4, 6)).astype(np.float32)},
        ids=np.arange(32, dtype=np.int64) % 2, codes=['a', 'b'], horizons=[1, 5, 21, 63, 126],
        target_scales=.01*np.sqrt(np.asarray([1, 5, 21, 63, 126], dtype=np.float32)),
        y=rng.normal(size=(32, 5))*.02)
    p = dict(architecture='tcn', learning_rate=.001, weight_decay=.001, gradient_clip=1., batch_size=16,
             max_epochs=2, minimum_epochs=2, patience=3)
    return data, p


class TargetTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_roundtrip_monotonicity_oddness_and_equal_local_slope(self):
        values = np.linspace(-10, 10, 1001)
        for spec in ({'kind': 'raw'}, {'kind': 'asinh', 'alpha': .5},
                     {'kind': 'asinh', 'alpha': 1.}, {'kind': 'signed_exp', 'alpha': .1},
                     {'kind': 'signed_exp', 'alpha': .25}):
            changed = transform(values, spec)
            self.assertTrue((np.diff(changed) > 0).all())
            np.testing.assert_allclose(changed, -transform(-values, spec), atol=1e-12)
            np.testing.assert_allclose(inverse(changed, spec), values, atol=1e-12)
            self.assertAlmostEqual(float(transform([1e-7], spec)[0]/1e-7), 1., places=6)
        self.assertLess(transform([4.], {'kind': 'asinh', 'alpha': 1.})[0], 4.)
        self.assertGreater(transform([4.], {'kind': 'signed_exp', 'alpha': .25})[0], 4.)
        for function in (transform, inverse):
            with self.assertRaises(ValueError): function([1.], {'kind': 'asinh', 'alpha': 0})
            with self.assertRaises(ValueError): function([1.], {'kind': 'bogus', 'alpha': 1})

    def test_raw_ablation_matches_existing_fitter_and_predictions(self):
        data, p = fixture()
        rows = np.arange(24)
        mask = np.ones((24, 5), dtype=bool)
        original, _ = neural_engine.fit(data, 'tcn', 20, 12, rows, mask, p, epochs=2)
        repeated, _ = transform_engine.fit(data, 20, 12, rows, mask, p, {'kind': 'raw'}, epochs=2)
        np.testing.assert_array_equal(neural_engine.predict(original, data, 20, rows),
                                     transform_engine.predict(repeated, data, 20, rows, {'kind': 'raw'}))

    def test_early_stopping_score_is_original_return_mae_after_inversion(self):
        data, p = fixture()
        train, valid = np.arange(24), np.arange(24, 32)
        tm, vm = np.ones((24, 5), dtype=bool), np.ones((8, 5), dtype=bool)
        spec = {'kind': 'asinh', 'alpha': 1.}
        net, log = transform_engine.fit(data, 20, 12, train, tm, p, spec, validation=(valid, vm))
        value = transform_engine.predict(net, data, 20, valid, spec)
        expected = neural_engine.scaled_score(data, valid, vm, value)
        self.assertAlmostEqual(log['history'][-1]['validation_original_scaled_mae'], expected, places=12)

    def test_family_strength_selection_uses_validation_only(self):
        candidates = ['raw', 'asinh_0.5', 'asinh_1', 'signed_exp_0.1', 'signed_exp_0.25']
        task = {'choice': 'raw', 'nonlinear_choice': 'asinh_0.5',
                'validation_mae_pp': dict(zip(candidates, [1., 1.1, 1.2, 1.3, 1.4])),
                'test_mae_pp': dict(zip(candidates, [5., 9., 0., 9., 0.]))}
        maps = selection_maps({('000300', 1): task}, candidates)
        self.assertEqual(maps['asinh_selected'][('000300', 1)], 'asinh_0.5')
        self.assertEqual(maps['signed_exp_selected'][('000300', 1)], 'signed_exp_0.1')
        self.assertEqual(maps['validation_selected'][('000300', 1)], 'raw')

    def test_unseen_and_masked_labels_cannot_change_training_and_reload_needs_transform(self):
        data, p = fixture()
        rows = np.arange(24)
        mask = np.ones((24, 5), dtype=bool)
        mask[:, -1] = False
        spec = {'kind': 'asinh', 'alpha': 1.}
        one, _ = transform_engine.fit(data, 20, 12, rows, mask, p, spec, epochs=2)
        expected = transform_engine.predict(one, data, 20, np.arange(24, 32), spec)
        data.y[24:] = 999.
        data.y[:24, -1] = np.nan
        two, _ = transform_engine.fit(data, 20, 12, rows, mask, p, spec, epochs=2)
        np.testing.assert_array_equal(expected, transform_engine.predict(two, data, 20, np.arange(24, 32), spec))
        with tempfile.TemporaryDirectory() as temp:
            bundle = transform_engine.export(two, temp, dict(architecture='tcn', sequence_length=4,
                channels=list(range(6)), codes=data.codes, horizons=data.horizons, target_transform=spec))
            restored, metadata = transform_engine.load(bundle)
            self.assertEqual(metadata['target_transform'], spec)
            np.testing.assert_array_equal(expected, transform_engine.predict(restored, data, 20, np.arange(24, 32), spec))
            with self.assertRaises(ValueError): neural_engine.load(bundle)
            weights = Path(temp)/'weights.safetensors'
            weights.write_bytes(weights.read_bytes()+b'bad')
            with self.assertRaises(ValueError): transform_engine.load(bundle)
