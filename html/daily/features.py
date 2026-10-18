"""Past-only features matching the frozen training feature definition."""
import math
import statistics

FEATURES=('momentum_5','momentum_21','momentum_63','momentum_126','momentum_252',
          'volatility_21','volatility_63','ma_gap_21','ma_gap_63','drawdown_252')
ONE_DAY_FEATURES=FEATURES+('momentum_1','momentum_2','momentum_3','momentum_10','volatility_5')


def features_at(values, i):
    close=values[i]
    row={f'momentum_{h}':close/values[i-h]-1 for h in (5,21,63,126,252)}
    for h in (21,63):
        returns=[math.log(values[j]/values[j-1]) for j in range(i-h+1,i+1)]
        row[f'volatility_{h}']=statistics.stdev(returns)*math.sqrt(252)
        row[f'ma_gap_{h}']=close/statistics.mean(values[i-h+1:i+1])-1
    row['drawdown_252']=close/max(values[i-251:i+1])-1
    return row


def one_day_features(values,i):
    row=features_at(values,i)
    for h in (1,2,3,10):
        row[f'momentum_{h}']=values[i]/values[i-h]-1
    returns=[math.log(values[j]/values[j-1]) for j in range(i-4,i+1)]
    row['volatility_5']=statistics.stdev(returns)*math.sqrt(252)
    return row
