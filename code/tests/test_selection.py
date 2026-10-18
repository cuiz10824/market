import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'html'))
from daily.selection import load_selection, scoped_protocol


class SelectionTests(unittest.TestCase):
    def test_training_and_runtime_share_exact_eight_indices_in_requested_order(self):
        expected=['000510','000300','000928','000933','000932','000930','000934','000935']
        selected=load_selection(ROOT)
        self.assertEqual([i['code'] for i in selected],expected)
        self.assertEqual([i['display_name'] for i in selected],
                         ['中证A500','沪深300','能源','医药','消费','制造','金融','科技'])
        for name in ('protocol.json','one_day_protocol.json'):
            original=json.loads((ROOT/'code/training'/name).read_text(encoding='utf-8'))
            scoped=scoped_protocol(ROOT,original)
            self.assertEqual(scoped['codes'],expected)
            self.assertEqual(len(original['codes']),14)  # Historical base protocol is immutable.
            self.assertEqual(scoped['horizons'],original['horizons'])

    def test_incomplete_training_protocol_cannot_silently_drop_selected_indices(self):
        with self.assertRaises(ValueError):scoped_protocol(ROOT,{'codes':['000300']})


if __name__=='__main__':unittest.main()
