"""Causality, temporal masking, network heads and checkpoint integrity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'html')]
from research.sequence_data import SequenceData,CHANNELS
from research.neural_models import SharedForecaster,CausalBlock
from training.neural_engine import export,load,masked_loss


def fixture():
    p={'codes':['000300','000510'],'horizons':[1,5,21,63,126],'lookbacks':[10,20,30]}
    calendar=pd.bdate_range('2010-01-04',periods=230).strftime('%Y-%m-%d').tolist();bars={}
    for j,code in enumerate(p['codes']):
        x=np.arange(len(calendar));close=100*np.exp(.002*x+.02*np.sin(.17*x+j))
        bars[code]=pd.DataFrame(dict(date=calendar,open=close*.999,high=close*1.01,low=close*.99,close=close,volume=1000+x))
    return p,bars,calendar


class NeuralTests(unittest.TestCase):
    def test_blocks_never_use_future_and_remain_invariant_to_price_volume_units(self):
        p,bars,calendar=fixture();d=SequenceData(p,bars,calendar)
        altered={c:f.copy() for c,f in bars.items()}
        for f in altered.values():f.loc[f.date>calendar[120],['open','high','low','close','volume']]*=2
        changed=SequenceData(p,altered,calendar);past=d.dates<=calendar[120]
        for length in p['lookbacks']:np.testing.assert_array_equal(d.x[length][past],changed.x[length][past])
        scaled={c:f.copy() for c,f in bars.items()}
        for f in scaled.values():
            f[['open','high','low','close']]*=10;f['volume']*=100
        units=SequenceData(p,scaled,calendar)
        for length in p['lookbacks']:np.testing.assert_allclose(d.x[length],units.x[length],atol=1e-6,rtol=1e-5)
        for length in p['lookbacks']:self.assertEqual(d.x[length].shape[1],(length+4)//5)

    def test_label_cutoff_is_enforced_per_head_and_future_gap_is_not_skipped(self):
        p,bars,calendar=fixture();bars['000300']=bars['000300'][bars['000300'].date!=calendar[90]]
        d=SequenceData(p,bars,calendar);cutoff=calendar[170]
        rows,mask=d.split(cutoff,train=True)
        self.assertTrue((d.label_indices[rows][mask]<170).all())
        self.assertTrue(((mask.sum(1)>0)&(mask.sum(1)<5)).any())
        for row in range(len(d.ids)):
            if d.ids[row]!=0:continue
            self.assertFalse(90<=d.ends[row]<=120)
            for j,h in enumerate(p['horizons']):
                if d.known[row,j]:self.assertFalse(d.ends[row]<=90<=d.ends[row]+h)
        # Masked future targets cannot affect optimization loss.
        out=torch.zeros(2,5);target=torch.ones(2,5);known=torch.ones(2,5);known[:,4]=0
        expected=masked_loss(out,target,known);target[:,4]=100000
        self.assertEqual(float(expected),float(masked_loss(out,target,known)))

    def test_all_architectures_have_five_heads_safe_reload_and_backward(self):
        for architecture in ('dlinear','nlinear','tcn','patchtst'):
            torch.manual_seed(2);net=SharedForecaster(architecture,51,6)
            x=torch.randn(3,51,6);identity=torch.tensor([0,3,7]);net.eval();value=net(x,identity)
            self.assertEqual(tuple(value.shape),(3,5));value.square().mean().backward()
            self.assertTrue(all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None))
            with tempfile.TemporaryDirectory() as temp:
                spec=dict(architecture=architecture,sequence_length=51,channels=list(CHANNELS),codes=[str(i) for i in range(8)],horizons=[1,5,21,63,126])
                path=export(net,temp,spec);restored,_=load(path)
                torch.testing.assert_close(value,restored(x,identity))
                weights=Path(temp)/'weights.safetensors';weights.write_bytes(weights.read_bytes()+b'bad')
                with self.assertRaises(ValueError):load(path)

    def test_tcn_causal_block_cannot_read_later_tokens(self):
        torch.manual_seed(1);block=CausalBlock(8,8).eval();x=torch.randn(2,8,80)
        original=block(x);x[:,:,50:]+=100
        torch.testing.assert_close(original[:,:,:50],block(x)[:,:,:50])


if __name__=='__main__':unittest.main()
