"""Past-only OHLCV, cross-index and market-state features shared by training/inference."""
import numpy as np
import pandas as pd

MOMENTUM=(1,2,3,5,10,21,63,126,252)
WINDOWS=(5,21,63)


def feature_names(codes):
    base=[f'momentum_{h}' for h in MOMENTUM]
    base += [f'daily_return_lag_{h}' for h in (1,2,3,5,10)]
    base += ['intraday','overnight','range','close_location','drawdown_252','volume_change_1']
    for h in WINDOWS:
        base += [f'volatility_{h}',f'ma_gap_{h}',f'up_fraction_{h}',f'volume_ratio_{h}',
                 f'range_mean_{h}',f'intraday_mean_{h}',f'overnight_mean_{h}']
    base += ['volatility_ratio','volume_return_corr_21','max_return_21','min_return_21',
             'breadth_1','breadth_21','dispersion_21','momentum_rank_21']
    for h in (1,5,21,63,126):base += [f'market_momentum_{h}',f'a500_momentum_{h}']
    base += ['market_volatility_21','market_drawdown_252']
    for h in (5,21,63):base += [f'relative_market_{h}']
    base += [f'peer_momentum_21_{code}' for code in codes]
    base += [f'index_{code}' for code in codes]
    return tuple(base)


def build_frames(bars,codes,calendar):
    frames={}
    for code in codes:
        frame=bars[code].copy().set_index('date').reindex(calendar)
        close=frame['close'];ret=close.pct_change(fill_method=None)
        feature=pd.DataFrame(index=calendar)
        for h in MOMENTUM:feature[f'momentum_{h}']=close/close.shift(h)-1
        for h in (1,2,3,5,10):feature[f'daily_return_lag_{h}']=ret.shift(h)
        feature['intraday']=close/frame.open-1
        feature['overnight']=frame.open/close.shift(1)-1
        feature['range']=frame.high/frame.low-1
        feature['close_location']=(close-frame.low)/(frame.high-frame.low).replace(0,np.nan)
        feature['drawdown_252']=close/close.rolling(252).max()-1
        feature['volume_change_1']=frame.volume/frame.volume.shift(1)-1
        for h in WINDOWS:
            feature[f'volatility_{h}']=np.log(close/close.shift(1)).rolling(h).std()*np.sqrt(252)
            feature[f'ma_gap_{h}']=close/close.rolling(h).mean()-1
            feature[f'up_fraction_{h}']=ret.gt(0).where(ret.notna()).rolling(h).mean()
            feature[f'volume_ratio_{h}']=frame.volume/frame.volume.rolling(h).mean()-1
            for name in ('range','intraday','overnight'):feature[f'{name}_mean_{h}']=feature[name].rolling(h).mean()
        feature['volatility_ratio']=feature.volatility_5/feature.volatility_63
        feature['volume_return_corr_21']=ret.rolling(21).corr(np.log(frame.volume/frame.volume.shift(1)))
        feature['max_return_21']=ret.rolling(21).max();feature['min_return_21']=ret.rolling(21).min()
        # A complete 253-session past window is mandatory; no fill across real gaps.
        feature.loc[close.rolling(253).count()!=253,:]=np.nan
        frames[code]=feature
    daily=pd.DataFrame({c:frames[c].momentum_1 for c in codes})
    monthly=pd.DataFrame({c:frames[c].momentum_21 for c in codes})
    complete=daily.notna().all(axis=1)&monthly.notna().all(axis=1)
    for code,f in frames.items():
        f['breadth_1']=daily.gt(0).mean(axis=1).where(complete)
        f['breadth_21']=monthly.gt(0).mean(axis=1).where(complete)
        f['dispersion_21']=monthly.std(axis=1).where(complete)
        f['momentum_rank_21']=monthly.rank(axis=1,pct=True)[code].where(complete)
        for h in (1,5,21,63,126):
            f[f'market_momentum_{h}']=frames['000300'][f'momentum_{h}']
            f[f'a500_momentum_{h}']=frames['000510'][f'momentum_{h}']
        f['market_volatility_21']=frames['000300'].volatility_21
        f['market_drawdown_252']=frames['000300'].drawdown_252
        for h in (5,21,63):f[f'relative_market_{h}']=f[f'momentum_{h}']-frames['000300'][f'momentum_{h}']
        for peer in codes:
            f[f'peer_momentum_21_{peer}']=frames[peer].momentum_21
            f[f'index_{peer}']=float(code==peer)
        f.replace([np.inf,-np.inf],np.nan,inplace=True)
        frames[code]=f.loc[:,list(feature_names(codes))]
    return frames
