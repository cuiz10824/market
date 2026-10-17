import hashlib
import io
import json
import struct
import unittest
import zipfile

from tools.market_data import parse_eastmoney, parse_tencent, qlib_rows, validate_rows


class DatasetTests(unittest.TestCase):
    item = {'code': '000300', 'name': '沪深300', 'secid': '1.000300'}

    def snapshot(self, candles, start='2025-01-02', end='2025-01-03'):
        raw = json.dumps({'code': 0, 'data': {'sh000300': {'day': candles}}})
        return {'symbol': 'sh000300', 'windows': [{
            'start': start, 'end': end, 'response_text': raw,
            'sha256': hashlib.sha256(raw.encode()).hexdigest()}]}

    def test_qlib_uses_adjclose_and_calendar_offset_not_rebased_close(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as z:
            z.writestr('calendars/day.txt', '2025-01-01\n2025-01-02\n2025-01-03\n')
            z.writestr('features/sh000300/close.day.bin', struct.pack('<fff', 1, 1, 1.1))
            z.writestr('features/sh000300/adjclose.day.bin', struct.pack('<fff', 1, 4000, 4400))
        with zipfile.ZipFile(stream) as z:
            rows = qlib_rows(z, 'sh000300', self.item)
        self.assertEqual([r['date'] for r in rows], ['2025-01-02', '2025-01-03'])
        self.assertEqual([r['close'] for r in rows], ['4000.00', '4400.00'])

    def test_tencent_filters_outside_window_and_preserves_source_prices(self):
        snapshot = self.snapshot([
            ['2024-12-31', '100', '101', '102', '99'],
            ['2025-01-02', '101', '102.10', '103', '100'],
            ['2025-01-03', '102', '99.50', '103', '99']])
        rows = parse_tencent(snapshot, self.item)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]['close'], '99.50')

    def test_tencent_rejects_tampered_raw_response(self):
        snapshot = self.snapshot([['2025-01-02', '100', '101', '102', '99']])
        snapshot['windows'][0]['response_text'] += ' '
        with self.assertRaisesRegex(ValueError, 'digest'):
            parse_tencent(snapshot, self.item)

    def test_adjusted_stock_series_cannot_replace_index(self):
        snapshot = self.snapshot([])
        payload = {'code': 0, 'data': {'sh000300': {'qfqday': [['2025-01-02', '100', '101', '102', '99']]}}}
        raw = json.dumps(payload)
        snapshot['windows'][0].update(response_text=raw, sha256=hashlib.sha256(raw.encode()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'Unadjusted'):
            parse_tencent(snapshot, self.item)

    def test_eastmoney_checks_identity_and_ohlc(self):
        payload = {'rc': 0, 'data': {'code': '000300', 'name': '沪深300',
                                   'klines': ['2025-01-02,100,101,102,99,1,1']}}
        self.assertEqual(parse_eastmoney(payload, self.item)[0]['close'], '101')
        payload['data']['name'] = '其他指数'
        with self.assertRaises(ValueError):
            parse_eastmoney(payload, self.item)
        payload['data']['name'] = '沪深300'
        payload['data']['klines'] = ['2025-01-02,100,101,100,99,1,1']
        with self.assertRaises(ValueError):
            parse_eastmoney(payload, self.item)

    def test_invalid_observation_dates_and_prices_fail_closed(self):
        for date, close in [('2025-01-04', '100'), ('2099-01-01', '100'),
                            ('2025-01-02', 'NaN'), ('2025-01-02', '0')]:
            with self.assertRaises(ValueError):
                validate_rows([{'date': date, 'close': close}])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_rows([{'date': '2025-01-02', 'close': '100'}] * 2)


if __name__ == '__main__':
    unittest.main()
