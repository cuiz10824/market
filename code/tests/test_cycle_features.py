"""Long lookback must be causal, continuous, and identical in training/inference."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html')]
from daily.cycle_features import build_frames,feature_names,LOOKBACK
from daily.rich_features import build_frames as short_frames,feature_names as short_names
from research.cycle_dataset import samples

CODES=['000510','000300','000928','000933','000932','000930','000934','000935']


def history():
    calendar=pd.bdate_range('2005-01-04',periods=1850).strftime('%Y-%m-%d').tolist()
    x=np.arange(len(calendar));bars={}
    for j,code in enumerate(CODES):
        close=100*np.exp(.0002*x+.08*np.sin(x*.013+j)+.02*np.sin(x*.17))
        bars[code]=pd.DataFrame(dict(date=calendar,open=close*.998,high=close*1.01,low=close*.99,
                                    close=close,volume=10000+x+200*np.sin(x*.02+j)))
    return bars,calendar


class CycleFeaturesTests(unittest.TestCase):
    def test_exact_five_year_returns_and_no_warmup_shortcuts(self):
        bars,calendar=history();features=build_frames(bars,CODES,calendar)
        self.assertEqual(len(feature_names(CODES)),142)
        old=short_frames(bars,CODES,calendar)
        for code in CODES:
            frame=features[code]
            pd.testing.assert_frame_equal(frame.loc[:,list(short_names(CODES))],old[code])
            self.assertTrue(pd.isna(frame.loc[calendar[1259],'cycle_return_1260']))
            self.assertTrue(np.isfinite(frame.loc[calendar[1260]].to_numpy()).all())
            close=bars[code].close
            for h in (252,504,756,1008,1260):
                self.assertAlmostEqual(frame.iloc[-1][f'cycle_return_{h}'],close.iloc[-1]/close.iloc[-1-h]-1)
            for q in range(20):
                self.assertAlmostEqual(frame.iloc[-1][f'quarter_return_lag_{q}'],close.iloc[-1-63*q]/close.iloc[-1-63*(q+1)]-1)

    def test_future_changes_and_truncation_do_not_change_past_five_year_features(self):
        bars,calendar=history();cutoff=1500
        expected=build_frames(bars,CODES,calendar)
        changed={c:f.copy() for c,f in bars.items()}
        for f in changed.values():f.loc[f.date>calendar[cutoff],['open','high','low','close','volume']]*=3
        actual=build_frames(changed,CODES,calendar)
        truncated=build_frames({c:f.iloc[:cutoff+1] for c,f in bars.items()},CODES,calendar[:cutoff+1])
        for code in CODES:
            pd.testing.assert_frame_equal(expected[code].iloc[:cutoff+1],actual[code].iloc[:cutoff+1])
            pd.testing.assert_frame_equal(expected[code].iloc[:cutoff+1],truncated[code])

    def test_missing_past_session_and_future_label_are_not_silently_skipped(self):
        bars,calendar=history();missing=calendar[1400]
        bars['000933']=bars['000933'][bars['000933'].date!=missing]
        datasets,features=samples(bars,CODES,calendar)
        self.assertTrue(pd.isna(features['000933'].loc[calendar[1700],'cycle_return_1260']))
        for h,frame in datasets.items():
            subset=frame[frame.code=='000933']
            self.assertFalse((subset.date>=missing).any())
            self.assertFalse(((subset.date<=missing)&(subset.label_end_date>=missing)).any())
            self.assertTrue((frame.date>=calendar[LOOKBACK]).all())
            positions={d:i for i,d in enumerate(calendar)}
            self.assertTrue(all(positions[r.label_end_date]-positions[r.date]==h for r in frame.itertuples()))


if __name__=='__main__':unittest.main()
