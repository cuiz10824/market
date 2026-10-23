"""Past-only OHLCV features and separately isolated forward trend outcomes."""
import numpy as np
import pandas as pd

NAMES={0:'熊市趋势',1:'震荡/混合',2:'牛市趋势'}


def features(frame):
    c=frame.close.astype(float);v=frame.volume.astype(float);r=np.log(c).diff()
    values={}
    for n in (5,20,60,120,250,500,750,1000,1250):
        values[f'return_{n}']=c/c.shift(n)-1
    for n in (20,60,120,250,500,1250):
        mean=c.rolling(n).mean()
        values[f'price_ma_{n}']=c/mean-1
        values[f'ma_slope_{n}']=mean/mean.shift(20)-1
    for n in (63,126,252,750,1250):
        values[f'drawdown_{n}']=c/c.rolling(n).max()-1
        values[f'from_low_{n}']=c/c.rolling(n).min()-1
    for n in (20,60,120,250):
        values[f'volatility_{n}']=r.rolling(n).std()
        values[f'up_fraction_{n}']=(r>0).astype(float).rolling(n).mean()
    for n in (5,20,60,120,250):
        values[f'volume_ratio_{n}']=v.rolling(n).mean()/v.rolling(250).mean().replace(0,np.nan)-1
    values['daily_range']=(frame.high-frame.low)/c
    values['range_20']=values['daily_range'].rolling(20).mean()
    values['body']=(c-frame.open)/frame.open
    values['gap']=frame.open/c.shift()-1
    return pd.DataFrame(values,index=frame.index).replace([np.inf,-np.inf],np.nan)


def targets(frame,horizon,threshold,occupancy=.6):
    close=frame.close.to_numpy(dtype=float);n=len(close)
    above=np.zeros(n);below=np.zeros(n);known=np.zeros(n,dtype=int)
    for offset in range(1,horizon+1):
        above[:-offset]+=close[offset:]>close[:-offset]
        below[:-offset]+=close[offset:]<close[:-offset]
        known[:-offset]+=1
    returns=frame.close.shift(-horizon)/frame.close-1
    labels=np.full(n,-1,dtype=int);valid=known==horizon
    labels[valid]=1
    labels[valid&(returns.to_numpy()>=threshold)&(above/horizon>=occupancy)]=2
    labels[valid&(returns.to_numpy()<=-threshold)&(below/horizon>=occupancy)]=0
    return pd.DataFrame(dict(label=labels,forward_return=returns,
                             label_end=frame.date.shift(-horizon),
                             above_fraction=np.where(valid,above/horizon,np.nan),
                             below_fraction=np.where(valid,below/horizon,np.nan)))


def ma_state(frame):
    close=frame.close;ma60=close.rolling(60).mean();ma120=close.rolling(120).mean();ma250=close.rolling(250).mean()
    slope=ma120/ma120.shift(20)-1
    bull=(close>ma250)&(ma60>ma120)&(slope>0)
    bear=(close<ma250)&(ma60<ma120)&(slope<0)
    return np.where(bull,2,np.where(bear,0,1))


def cycles(frame,reversal=.2):
    """HINDSIGHT directional-change legs; confirmation is later than each pivot."""
    close=frame.close.to_numpy();dates=frame.date.to_numpy();result=[]
    low=high=0;direction=0;start=0
    for i in range(1,len(close)):
        if direction==0:
            if close[i]<close[low]:low=i
            if close[i]>close[high]:high=i
            if close[i]/close[low]-1>=reversal:direction=1;start=low;high=i
            elif close[i]/close[high]-1<=-reversal:direction=-1;start=high;low=i
        elif direction==1:
            if close[i]>close[high]:high=i
            if close[i]/close[high]-1<=-reversal:
                result.append(dict(start=dates[start],end=dates[high],direction='bull',
                    change=float(close[high]/close[start]-1),confirmed_at=dates[i],confirmed=True))
                direction=-1;start=high;low=i
        else:
            if close[i]<close[low]:low=i
            if close[i]/close[low]-1>=reversal:
                result.append(dict(start=dates[start],end=dates[low],direction='bear',
                    change=float(close[low]/close[start]-1),confirmed_at=dates[i],confirmed=True))
                direction=1;start=low;high=i
    if direction:
        result.append(dict(start=dates[start],end=dates[-1],direction='bull' if direction==1 else 'bear',
            change=float(close[-1]/close[start]-1),confirmed_at=None,confirmed=False))
    return pd.DataFrame(result)


def split(frame,x,target,start,end=None,train=False):
    ready=x.notna().all(axis=1)
    if train:
        selected=ready&(target.label>=0)
        if start:selected &= target.label_end<start
    else:
        selected=ready&(target.label>=0)&(frame.date>=start)
        if end:selected &= (frame.date<end)&(target.label_end<end)
    return np.flatnonzero(selected.to_numpy())
