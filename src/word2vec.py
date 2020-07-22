import gensim
import pickle
from gensim.models import Word2Vec
from multiprocessing import cpu_count
import pandas as pd
import numpy as np

path_train = '../data/semi/folds/{}/{}'
path_test = '../data/precompetition/{}/{}'


def train_w2v(latent_dim=200):
    for col in ['time', 'creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
        # for col in ['time', ]:
        print(col)
        N = None
        corpus = pd.concat([
            pd.read_csv(path_train.format('fold_1', 'origin_feature_reindexed_fromzero.csv'), usecols=[col], nrows=N),
            pd.read_csv(path_train.format('fold_2', 'origin_feature_reindexed_fromzero.csv'), usecols=[col], nrows=N),
            pd.read_csv(path_train.format('fold_3', 'origin_feature_reindexed_fromzero.csv'), usecols=[col], nrows=N),
            pd.read_csv(path_train.format('fold_4', 'origin_feature_reindexed_fromzero.csv'), usecols=[col], nrows=N),
            pd.read_csv(path_train.format('fold_5', 'origin_feature_reindexed_fromzero.csv'), usecols=[col], nrows=N),
            pd.read_csv(path_test.format('test_preliminary', 'origin_feature_reindexed_fromzero.csv'), usecols=[col],
                        nrows=N),
        ], axis=0)

        corpus = list(corpus[col].values)
        corpus = [item.split(' ') for item in corpus]
        print(corpus[0], corpus[-1])

        # model train
        model = Word2Vec(corpus, min_count=0, workers=36, size=latent_dim, window=9, sg=1, hs=1, iter=10)
        model.save('../models/w2v/model_{}_p20200628'.format(col))
        del corpus, model


def get_w2v_embedding_npy(latent_dim=200):
    for col in ['time', 'creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
        print(col)
        model = Word2Vec.load('../models/w2v/model_{}_p20200628'.format(col))
        key_list = sorted(list(model.wv.vocab), key=lambda x: len(model.wv.vocab) if x == '\\N' else int(x))
        print(key_list[:5], key_list[-5:], len(key_list))

        res = np.zeros((max([int(x) for x in key_list]) + 1, latent_dim), dtype=np.float16)

        if '0' not in key_list:
            tmp = np.zeros((latent_dim,))
            for key in key_list:
                tmp += np.array(model.wv[key])
            tmp /= len(key_list)
            res[0] = tmp

        print(res.shape)
        for i in range(len(key_list)):
            if i % 100000 == 0:
                print(i)
            res[int(key_list[i])] = model.wv[key_list[i]].astype(np.float16)

        np.save("../models/w2v/w2v_embeddings_{}_p20200628.npy".format(col), res)


if __name__ == '__main__':
    train_w2v(200)
    get_w2v_embedding_npy(200)