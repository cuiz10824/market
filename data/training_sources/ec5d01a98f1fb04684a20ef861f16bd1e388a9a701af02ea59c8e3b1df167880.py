"""Next observed trading session labels; all features end at the current close."""
import math
import statistics

from research.dataset import FEATURES as BASE_FEATURES, aligned_prices, features_at

FEATURES=BASE_FEATURES+('momentum_1','momentum_2','momentum_3','momentum_10','volatility_5')


def one_day_features(values,i):
    row=features_at(values,i)
    for h in (1,2,3,10):
        row[f'momentum_{h}']=values[i]/values[i-h]-1
    returns=[math.log(values[j]/values[j-1]) for j in range(i-4,i+1)]
    row['volatility_5']=statistics.stdev(returns)*math.sqrt(252)
    return row


def build_one_day_samples(calendar,prices,code=None):
    values,missing=aligned_prices(calendar,prices)
    rows=[]
    for i in range(252,len(calendar)-1):
        if missing[i+2]!=missing[i-252]:
            continue
        target=values[i+1]/values[i]-1
        rows.append(dict(date=calendar[i],label_end_date=calendar[i+1],horizon=1,code=code,
                         target_return=target,**one_day_features(values,i)))
    return rows
