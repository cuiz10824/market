"""A forced approval is an isolated fixture, never a production selection."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'html'),str(ROOT/'code')]
from daily.enhanced_runtime import EnhancedRuntime
from daily.enhanced_predictor import EnhancedPredictor
from daily.cycle_features import build_frames
from daily.service import DailyService


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')


class CycleRuntimeTests(unittest.TestCase):
    def test_five_year_saved_model_is_applied_and_missing_history_falls_back(self):
        pointer=read(ROOT/'code/models/latest_five_year.json');index=read(ROOT/pointer['index'])
        task=copy.deepcopy(next(t for t in index['tasks'] if t['code']=='000300' and t['horizon']==1))
        bundle=next(m['bundle'] for m in index['models'] if m['role']=='latest' and m['horizon']==1 and m['candidate']=='lightgbm_mae_5y')
        task.update(choice='lightgbm_mae_5y',latest_bundle=bundle)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'code/models').mkdir(parents=True);(root/'data/enhanced').mkdir(parents=True)
            (root/'code/models'/pointer['run_id']).symlink_to((ROOT/pointer['index']).parent,target_is_directory=True)
            (root/'data/enhanced/market_ohlcv.sqlite3').symlink_to(ROOT/'data/enhanced/market_ohlcv.sqlite3')
            write(root/'fixture-index.json',dict(index,tasks=[task]))
            write(root/'fixture-verification.json',read(ROOT/pointer['verification']))
            write(root/'code/models/latest_five_year.json',dict(pointer,index='fixture-index.json',verification='fixture-verification.json'))
            service=DailyService(store=root/'daily.sqlite3')
            codes=[item['code'] for item in service.items]
            runtime=EnhancedRuntime(root,codes,store=root/'rich.sqlite3',pointer_name='latest_five_year.json')
            rows=copy.deepcopy(service.payload['rows']);date=next(r['as_of'] for r in rows if r['code']=='000300')
            features=build_frames(runtime.bars,codes,service.calendar)['000300'].loc[date].to_dict()
            expected=float(EnhancedPredictor(ROOT/bundle).predict([features])[0])*100
            result=runtime.apply(rows,service.calendar)
            row=next(r for r in rows if r['code']=='000300')
            self.assertAlmostEqual(row['forecast']['1'],expected,places=12)
            self.assertIn('五年',row['models']['1']);self.assertEqual(result['upgraded_tasks'],1)
            self.assertEqual(result['lookback_sessions'],1260)
            # Corrupt only the temporary in-memory history; preserve original files.
            frame=runtime.bars['000300'];runtime.bars['000300']=frame.drop(frame.index[-400])
            rows=copy.deepcopy(service.payload['rows']);before=next(r['forecast']['1'] for r in rows if r['code']=='000300')
            result=runtime.apply(rows,service.calendar);row=next(r for r in rows if r['code']=='000300')
            self.assertEqual(row['forecast']['1'],before)
            self.assertIn('保留旧模型',row['enhancement_error']);self.assertEqual(result['upgraded_tasks'],0)


if __name__=='__main__':unittest.main()
