import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bear.dataset import features,targets,split,cycles
from bear.models import fit,predict,save,load
from collect import parse,SHANGHAI


def fixture(n=1600):
    t=np.arange(n);close=100*np.exp(.0001*t+.15*np.sin(t/80))
    return pd.DataFrame(dict(date=pd.bdate_range('2000-01-03',periods=n).strftime('%Y-%m-%d'),
                            open=close*.999,high=close*1.01,low=close*.99,close=close,volume=1000+t))


class BearTests(unittest.TestCase):
    def test_features_never_read_future_and_are_invariant_to_price_volume_units(self):
        frame=fixture();changed=frame.copy()
        changed.loc[1500:,['open','high','low','close','volume']]*=3
        original=features(frame)
        pd.testing.assert_frame_equal(original.iloc[:1500],features(changed).iloc[:1500])
        units=frame.copy();units[['open','high','low','close']]*=10;units.volume*=100
        np.testing.assert_allclose(features(units).iloc[1270:],original.iloc[1270:],rtol=1e-8,atol=1e-10)

    def test_forward_labels_are_unknown_until_complete_and_use_path(self):
        frame=fixture(30);frame.close=np.linspace(100,160,30)
        target=targets(frame,10,.06,.6)
        self.assertEqual(target.label.iloc[0],2)
        self.assertTrue((target.label.tail(10)==-1).all())
        frame.close=np.linspace(160,100,30)
        self.assertEqual(targets(frame,10,.06,.6).label.iloc[0],0)
        frame.close=100.
        self.assertEqual(targets(frame,10,.06,.6).label.iloc[0],1)
        frame.close=[100]+[90]*9+[110]+[110]*19
        self.assertEqual(targets(frame,10,.06,.6).label.iloc[0],1)

    def test_label_end_is_purged_and_persistence_is_causal(self):
        frame=fixture();x=features(frame);target=targets(frame,63,.06)
        cutoff=frame.date.iloc[1500]
        rows=split(frame,x,target,cutoff,train=True)
        self.assertTrue((target.label_end.iloc[rows]<cutoff).all())
        changed=frame.copy();changed.loc[1501:,'close']*=3
        new=targets(changed,63,.06)
        np.testing.assert_array_equal(target.label.shift(63).iloc[1270:1501],new.label.shift(63).iloc[1270:1501])

    def test_cycles_are_hindsight_with_confirmation_and_last_leg_unconfirmed(self):
        frame=fixture(8);frame.close=[100,120,150,140,115,100,110,125]
        legs=cycles(frame,.2)
        self.assertEqual(legs.direction.iloc[:2].tolist(),['bull','bear'])
        self.assertTrue((legs.loc[legs.confirmed,'confirmed_at']>legs.loc[legs.confirmed,'end']).all())
        self.assertFalse(bool(legs.confirmed.iloc[-1]))

    def test_index_identity_and_partial_today_are_validated(self):
        payload={'code':0,'data':{'sh000001':{'qt':{'sh000001':['1','上证指数','000001']},
            'day':[['2026-10-21','100','101','102','99','1000'],['2026-10-22','101','102','103','100','1100']]}}}
        self.assertEqual(len(parse(payload,dt.datetime(2026,10,22,12,tzinfo=SHANGHAI))),1)
        payload['data']['sh000001']['qt']['sh000001'][1]='平安银行'
        with self.assertRaises(ValueError):parse(payload,dt.datetime(2026,10,22,16,tzinfo=SHANGHAI))

    def test_native_models_reload_without_pickle(self):
        rng=np.random.default_rng(2)
        x=pd.DataFrame(rng.normal(size=(600,4)),columns=list('abcd'));y=np.tile([0,1,2],200)
        for kind in ('majority','persistence','ma_rule','logistic','lightgbm'):
            model=fit(kind,x,y,dict(seed=1,threads=2))
            expected=predict(model,x,y,y)
            with tempfile.TemporaryDirectory() as tmp:
                path=save(model,tmp,dict(features=list(x.columns)))
                np.testing.assert_allclose(predict(load(path),x,y,y),expected,atol=1e-12)
                if kind=='lightgbm':
                    native=Path(tmp)/'model.lgb.txt';native.write_bytes(native.read_bytes()+b'bad')
                    with self.assertRaises(ValueError):load(path)
