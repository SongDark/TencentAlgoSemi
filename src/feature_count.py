import pandas as pd
import pickle
from collections import Counter

path = '../data/{stage}/{phase}/{file}'

def feat_count():
    the_dict = dict()
    for key in ['time', 'creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
        N = None
        df = \
            pd.concat([
                pd.read_csv(path.format(stage='semi', phase='folds/fold_1', file='origin_feature.csv'), usecols=[key], nrows=N),
                pd.read_csv(path.format(stage='semi', phase='folds/fold_2', file='origin_feature.csv'), usecols=[key], nrows=N),
                pd.read_csv(path.format(stage='semi', phase='folds/fold_3', file='origin_feature.csv'), usecols=[key], nrows=N),
                pd.read_csv(path.format(stage='semi', phase='folds/fold_4', file='origin_feature.csv'), usecols=[key], nrows=N),
                pd.read_csv(path.format(stage='semi', phase='folds/fold_5', file='origin_feature.csv'), usecols=[key], nrows=N),
                pd.read_csv(path.format(stage='precompetition', phase='test_preliminary', file='origin_feature.csv'), usecols=[key], nrows=N)
            ], axis=0)
        print(df.shape)
        cnt = dict(Counter(" ".join(df[key].values).split(' ')))
        print(key, Counter(cnt.values())[1])
        
        the_dict.update({key + k : v for k, v in cnt.items()})
        
    with open(path.format(stage='semi', phase='dict', file='feature_cnt.pkl'), "wb") as fout:
        pickle.dump(the_dict, fout)

if __name__ == '__main__':
    feat_count()