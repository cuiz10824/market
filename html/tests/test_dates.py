import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from daily.dates import annotate_dates, target_date
from daily.page import table


class DateTests(unittest.TestCase):
    def test_exchange_holidays_and_weekend_makeup_days_are_not_sessions(self):
        self.assertEqual(target_date('2026-09-30', 1, set()), ('2026-10-08', False))
        self.assertEqual(target_date('2026-02-13', 1, set()), ('2026-02-24', False))
        self.assertEqual(target_date('2026-10-16', 1, set()), ('2026-10-19', False))
        self.assertEqual(target_date('2026-10-19', 1, set()), ('2026-10-20', False))

    def test_observed_calendar_overrides_estimate_and_unknown_year_is_marked(self):
        self.assertEqual(target_date('2027-01-01', 1, {'2027-01-01', '2027-01-05'}), ('2027-01-05', False))
        self.assertEqual(target_date('2026-12-31', 1, set()), ('2027-01-01', True))

    def test_actual_week_and_stale_prediction_keep_separate_dates(self):
        calendar=['2026-10-09', '2026-10-12', '2026-10-13', '2026-10-14', '2026-10-15', '2026-10-16', '2026-10-19']
        row={'as_of': '2026-10-19', 'actual': {'1': 1., '5': 2.}, 'forecast': {'1': .2, '5': None},
             'forecast_as_of': {'1': '2026-10-16', '5': None}}
        headers=annotate_dates([row], [1, 5], calendar)
        self.assertEqual(row['date_ranges']['actual']['5']['start'], '2026-10-12')
        self.assertEqual(headers['forecast']['1'], ['基于 2026-10-16', '至 2026-10-19'])
        self.assertIsNone(row['date_ranges']['forecast']['5']['end'])
        row['as_of']='2026-10-16'
        annotate_dates([row], [1, 5], calendar)
        self.assertEqual(row['date_ranges']['actual']['5']['start'], '2026-10-09')

    def test_snapshot_dates_and_no_script_fallback_agree(self):
        root=Path(__file__).resolve().parents[2]
        snapshot=(root/'html/snapshot.js').read_text()
        payload=json.loads(snapshot.split(' = ', 1)[1].strip().rstrip(';'))
        for row in payload['rows']:
            for h in payload['horizons']:
                self.assertEqual(row['date_ranges']['actual'][str(h)]['end'], row['as_of'])
                self.assertEqual(row['date_ranges']['forecast'][str(h)]['start'], row['forecast_as_of'][str(h)])
        output=table(payload, 'actual')
        for line in payload['date_headers']['actual']['5']:
            self.assertIn(line, output)
        self.assertIn('column-date', output)

    def test_mixed_dates_are_visible_in_cells(self):
        rows=[{'code': str(i), 'name': '测试', 'category': '测试', 'as_of': d,
               'actual': {'1': 1.}, 'forecast': {'1': .2}, 'models': {'1': 'TCN'}}
              for i,d in enumerate(['2026-10-16', '2026-10-19'])]
        headers=annotate_dates(rows, [1], ['2026-10-15', '2026-10-16', '2026-10-19'])
        self.assertEqual(headers['forecast']['1'], ['各指数日期不同', '详见各行'])
        output=table({'rows': rows, 'horizons': [1], 'date_headers': headers}, 'forecast')
        self.assertIn('至 2026-10-19', output)
        self.assertIn('至 2026-10-20', output)
