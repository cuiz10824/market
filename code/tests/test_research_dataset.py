import datetime as dt
import unittest

from research.dataset import FEATURES, build_samples, sample_indices, walk_forward


def history(n=1000):
    calendar = []
    date = dt.date(2010, 1, 1)
    while len(calendar) < n:
        if date.weekday() < 5:
            calendar.append(date.isoformat())
        date += dt.timedelta(days=1)
    return calendar, {day: 100 + i for i, day in enumerate(calendar)}


class ResearchDatasetTests(unittest.TestCase):
    def test_all_horizons_use_exact_calendar_end_and_drop_unknown_labels(self):
        calendar, prices = history(700)
        for horizon in (5, 21, 63, 126):
            samples = build_samples(calendar, prices, horizon)
            self.assertEqual(len(samples), 700 - 252 - horizon)
            self.assertEqual(samples[0]['date'], calendar[252])
            self.assertEqual(samples[0]['label_end_date'], calendar[252 + horizon])
            self.assertAlmostEqual(samples[0]['target_return'], (352 + horizon) / 352 - 1)
            self.assertEqual(samples[-1]['label_end_date'], calendar[-1])

    def test_features_do_not_change_when_future_prices_change(self):
        calendar, prices = history(400)
        initial = build_samples(calendar, prices, 21)[0]
        changed = {day: price * (2 if day > calendar[252] else 1) for day, price in prices.items()}
        altered = build_samples(calendar, changed, 21)[0]
        for feature in FEATURES:
            self.assertEqual(initial[feature], altered[feature])
        self.assertNotEqual(initial['target_return'], altered['target_return'])

    def test_missing_calendar_day_cannot_be_hidden_by_row_offsets(self):
        calendar, prices = history(800)
        del prices[calendar[270]]
        _, indices = sample_indices(calendar, prices, 21)
        self.assertTrue(indices)
        self.assertTrue(all(not (i - 252 <= 270 <= i + 21) for i in indices))
        self.assertNotIn(252, indices)

    def test_splits_purge_labels_by_end_date_not_only_training_rows(self):
        calendar, prices = history(1300)
        samples = build_samples(calendar, prices, 126)
        folds = list(walk_forward(samples, min_train=100, test_size=126))
        self.assertTrue(folds)
        for train, test in folds:
            self.assertGreaterEqual(len(train), 100)
            self.assertLess(max(r['label_end_date'] for r in train), test[0]['date'])
            self.assertTrue(set(r['date'] for r in train).isdisjoint(r['date'] for r in test))

    def test_mixed_horizons_and_duplicate_dates_are_rejected(self):
        calendar, prices = history(500)
        samples = build_samples(calendar, prices, 21)
        samples[1]['horizon'] = 126
        with self.assertRaisesRegex(ValueError, 'one forecast horizon'):
            list(walk_forward(samples))
        samples[1]['horizon'] = 21
        samples[0]['code'] = '000300'
        samples[1]['code'] = '000510'
        with self.assertRaisesRegex(ValueError, 'one index code'):
            list(walk_forward(samples))
        samples[1]['date'] = samples[0]['date']
        with self.assertRaisesRegex(ValueError, 'unique'):
            list(walk_forward(samples))


if __name__ == '__main__':
    unittest.main()
