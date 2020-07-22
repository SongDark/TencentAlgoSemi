# encoding:utf-8
# 按照词频重新排序

import pandas as pd 
import numpy as np 
import pickle

path = '../data/{stage}/{phase}/{file}'

def reindex_fromzero_semi(stage, phase):
    with open(path.format(stage=stage, phase='dict', file='feature_cnt.pkl'), "rb") as fin:
        the_dict = pickle.load(fin)
        the_key = sorted([k for k, v in the_dict.items() if v >= 2])
        del the_dict

        cols = ['creative_id', 'time', 'product_id', 'product_category', 'advertiser_id', 'industry', 'ad_id']
        tmp_dict = {col:[] for col in cols}
        for key in the_key:
            for col in cols:
                if col in key:
                    tmp_dict[col].append(key) 
        for col in tmp_dict:
            # "\\N" 就是1
            tmp_dict[col] = sorted(tmp_dict[col], key=lambda x: -1 if x[len(col):]=="\\N" else int(x[len(col):]))
            tmp_dict[col] = dict(zip(tmp_dict[col], np.arange(len(tmp_dict[col])).astype(int) + 1))  # 编号是从1到N的
        the_dict = {}
        for col in tmp_dict:
            the_dict.update(tmp_dict[col])

    data = pd.read_csv(path.format(stage=stage, phase=phase, file='origin_feature.csv'), nrows=None)

    for key in ['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
        print(key)
        data[key] = data.apply(lambda x: ' '.join([str(the_dict.get(key + v, 0)) for v in x[key].split(" ")]), axis=1)  # 只出现过一次的id会编码为0
        
    data.to_csv(path.format(stage=stage, phase=phase, file='origin_feature_reindexed_fromzero.csv'), index=False)


if __name__ == '__main__':

    # 训练集
    for i in range(5):
        reindex_fromzero_semi(stage='semi', phase='folds/fold_%d' % (i+1))
    # 测试集
    reindex_fromzero_semi(stage='semi', phase='test_preliminary')
