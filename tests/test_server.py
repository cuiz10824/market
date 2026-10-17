import unittest
from server import parse_history


class HistoryTests(unittest.TestCase):
    def payload(self):
        import datetime
        start = datetime.date(2025, 1, 1)
        return {'data': {'code': '000300', 'klines': [
            f'{start + datetime.timedelta(days=i)},100,{100+i},101,99' for i in range(130)]}}

    def test_parse_close_not_open(self):
        rows = parse_history(self.payload(), '沪深300', '000300')
        self.assertEqual(len(rows), 130)
        self.assertEqual(rows[-1]['close'], 229)

    def test_wrong_symbol_and_short_history(self):
        with self.assertRaises(ValueError):
            parse_history(self.payload(), '能源', '000928')
        p = self.payload()
        p['data']['klines'] = p['data']['klines'][:10]
        with self.assertRaises(ValueError):
            parse_history(p, '沪深300', '000300')

    def test_bad_prices(self):
        for value in ('nan', 'inf', '-1'):
            p = self.payload()
            p['data']['klines'][0] = f'2025-01-01,100,{value},101,99'
            with self.assertRaises(ValueError):
                parse_history(p, '沪深300', '000300')


if __name__ == '__main__':
    unittest.main()
