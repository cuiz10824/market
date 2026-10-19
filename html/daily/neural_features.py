"""Daily causal OHLCV blocks; exact pack function copied from frozen sequence_data.py."""
import numpy as np

CHANNELS=('log_close_vs_origin_close','opening_gap','high_low_range','intrabar_return','relative_daily_volume','block_return')
SCALES=np.array([1.,.02,.05,.02,1.,.03],dtype=np.float64)


def pack(raw,ends,lookback):
    """Each origin reads exactly lookback completed daily bars and their preceding close."""
    width=(lookback+4)//5
    finish=ends[:,None]-5*np.arange(width-1,-1,-1)[None,:]
    begin=np.maximum(finish-4,ends[:,None]-lookback+1)
    high=np.full(finish.shape,-np.inf);low=np.full(finish.shape,np.inf);volume=np.zeros(finish.shape)
    for lag in range(5):
        ix=finish-lag;valid=ix>=begin
        high=np.maximum(high,np.where(valid,raw[ix,1],-np.inf))
        low=np.minimum(low,np.where(valid,raw[ix,2],np.inf))
        volume+=np.where(valid,raw[ix,4],0.)
    opening=raw[begin,0];close=raw[finish,3];prior=raw[begin-1,3]
    cumulative=np.r_[0.,np.cumsum(np.nan_to_num(raw[:,4]))]
    mean_volume=(cumulative[ends+1]-cumulative[ends-lookback+1])/lookback
    daily_volume=volume/(finish-begin+1)
    x=np.stack([np.log(close/raw[ends,3,None]),np.log(opening/prior),np.log(high/low),
                np.log(close/opening),np.log(daily_volume/mean_volume[:,None]),np.log(close/prior)],axis=-1)
    x=(x/SCALES).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite sequence inputs')
    return x
