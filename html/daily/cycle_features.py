"""Five-year causal history, extending the immutable 252-session feature version."""
import numpy as np
import pandas as pd

from daily.rich_features import build_frames as short_frames,feature_names as short_names

LOOKBACK=1260
LONG_WINDOWS=(252,504,756,1008,1260)


def feature_names(codes):
    names=list(short_names(codes))
    for h in LONG_WINDOWS:
        names += [f'cycle_return_{h}',f'cycle_volatility_{h}',f'cycle_ma_gap_{h}',
                  f'cycle_drawdown_{h}',f'cycle_range_position_{h}',f'cycle_volume_ratio_{h}']
    names += [f'quarter_return_lag_{i}' for i in range(20)]
    for h in (504,756,1008,1260):
        names += [f'cycle_market_return_{h}',f'cycle_a500_return_{h}',f'cycle_relative_market_{h}']
    return tuple(names)


def build_frames(bars,codes,calendar):
    short=short_frames(bars,codes,calendar);long={}
    for code in codes:
        raw=bars[code].set_index('date').reindex(calendar)
        close=raw.close;ret=np.log(close/close.shift(1));f=pd.DataFrame(index=calendar)
        for h in LONG_WINDOWS:
            highest=close.rolling(h).max();lowest=close.rolling(h).min()
            f[f'cycle_return_{h}']=close/close.shift(h)-1
            f[f'cycle_volatility_{h}']=ret.rolling(h).std()*np.sqrt(252)
            f[f'cycle_ma_gap_{h}']=close/close.rolling(h).mean()-1
            f[f'cycle_drawdown_{h}']=close/highest-1
            f[f'cycle_range_position_{h}']=(close-lowest)/(highest-lowest).replace(0,np.nan)
            f[f'cycle_volume_ratio_{h}']=raw.volume.rolling(21).mean()/raw.volume.rolling(h).mean()-1
        for i in range(20):f[f'quarter_return_lag_{i}']=close.shift(63*i)/close.shift(63*(i+1))-1
        # Every one of the 1261 sessions must be observed, including volume and OHLC.
        complete=raw[['open','high','low','close','volume']].notna().all(axis=1).astype(int).rolling(LOOKBACK+1).sum()==LOOKBACK+1
        f.loc[~complete,:]=np.nan
        long[code]=f
    result={}
    for code in codes:
        f=pd.concat([short[code],long[code]],axis=1)
        for h in (504,756,1008,1260):
            f[f'cycle_market_return_{h}']=long['000300'][f'cycle_return_{h}']
            f[f'cycle_a500_return_{h}']=long['000510'][f'cycle_return_{h}']
            f[f'cycle_relative_market_{h}']=long[code][f'cycle_return_{h}']-long['000300'][f'cycle_return_{h}']
        result[code]=f.loc[:,list(feature_names(codes))].replace([np.inf,-np.inf],np.nan)
    return result
