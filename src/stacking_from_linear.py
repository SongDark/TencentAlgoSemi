import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import tqdm 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.sparse import save_npz, load_npz


def my_accuracy_score(y_true, y_pred):
    return np.sum(np.equal(y_true, y_pred).astype(int)) / float(len(y_true))

def train_sklearn_model(model_name, clf, train_feat, score, split_ids):
    print("running %s" % model_name)
    
    for i in range(len(split_ids)):
        print(i)
        tr = np.concatenate([split_ids[j] for j in range(len(split_ids)) if j!=i])
        va = split_ids[i]
        clf.fit(train_feat[tr], score[tr])
        joblib.dump(clf, '../models/sklearn_linear/{}_fold{}.model'.format(model_name, i))


def predict_sklearn_model(model_name, train_feat, score, test_feat, split_ids):
    print("predicting %s" % model_name)

    pred_train = list()
    pred_test = None 

    for i in range(len(split_ids)):
        print(i)
        va = split_ids[i]
        clf = joblib.load('../models/sklearn_linear/{}_fold{}.model'.format(model_name, i))

        pred_train.append(clf._predict_proba_lr(train_feat[va]).astype(np.float32))

        acc = accuracy_score(score[va], np.argmax(pred_train[-1], 1))
        print(acc)

        if i == 0:
            pred_test = clf._predict_proba_lr(test_feat).astype(np.float32)
        else:
            pred_test += clf._predict_proba_lr(test_feat).astype(np.float32)
    
    pred_test /= len(split_ids) 

    columns = ['{}_{}'.format(model_name, i) for i in range(pred_test.shape[1])]

    # return pred_train, pred_test
    return [pd.DataFrame(x, columns=columns, dtype=np.float32) for x in pred_train], pd.DataFrame(pred_test, columns=columns, dtype=np.float32)







path = '../data/semi/{}/{}'
cols = ['ad_id', 'product_id', 'advertiser_id', 'industry', 'product_category', 'creative_id']

#### 提取tfidf特征
nrows = None
for col in cols:
    print("[tfidf feat] %s" % col)

    # load all click data
    train_valid_click_log = pd.DataFrame()
    data_size = list()
    for i in range(5):
        df = pd.read_csv(path.format('folds/fold_%d' % (i+1), 'origin_feature_reindexed_fromzero.csv'), usecols=[col, 'age', 'gender'], nrows=nrows)
        train_valid_click_log = pd.concat([train_valid_click_log, df], 0)
        data_size.append(df.shape[0])
    age, gender = train_valid_click_log.age.values - 1, train_valid_click_log.gender.values - 1
    
    test_click_log = pd.read_csv(path.format('test_preliminary', 'origin_feature_reindexed_fromzero.csv'), usecols=[col], nrows=nrows)

    full_click_log = pd.concat([train_valid_click_log[[col]], test_click_log], 0)
    del train_valid_click_log, test_click_log
    
    tf = TfidfVectorizer(ngram_range=(1, 1))
    discuss_tf = tf.fit_transform(full_click_log[col]).astype(np.float32).tocsr()
    del full_click_log

    save_npz(path.format('test_preliminary', '%s_tfvec_csr.npz' % col), discuss_tf)


#### 训练 sklearn 线性模型
for col in cols:
    nrows = None
    train_valid_click_log = pd.DataFrame()
    data_size = list()
    for i in range(5):
        df = pd.read_csv(path.format('folds/fold_%d' % (i+1), 'origin_feature_reindexed_fromzero.csv'), usecols=['age', 'gender'], nrows=nrows)
        train_valid_click_log = pd.concat([train_valid_click_log, df], 0)
        data_size.append(df.shape[0])
    age, gender = train_valid_click_log.age.values - 1, train_valid_click_log.gender.values - 1
    
    discuss_tf = load_npz(path.format('test_preliminary', '%s_tfvec_csr.npz' % col))
    print(discuss_tf.dtype)

    train_feature = discuss_tf[:sum(data_size)]
    print(discuss_tf.shape, train_feature.shape)
    del discuss_tf

    train_ids = list()
    pos = 0
    for i in range(len(data_size)):
        train_ids.append(np.arange(data_size[i]) + pos)
        pos += data_size[i]
    print([(min(x), max(x)) for x in train_ids])
    
    model_list = [
        ['LogisticRegression', LogisticRegression(random_state=2020, C=3)],
        ['SGDClassifier', SGDClassifier(random_state=2020, loss='log')],
        ['PassiveAggressiveClassifier', PassiveAggressiveClassifier(random_state=2020, C=2)],
        # ['RidgeClassifier', RidgeClassifier(random_state=2020)], # 太慢了
        ['LinearSVC', LinearSVC(random_state=2020)]
    ]

    feat = pd.DataFrame()
    for m in model_list:
        train_sklearn_model(col + '_' + m[0], m[1], train_feature, age, train_ids)
        train_sklearn_model(col + '_' + m[0] + '_gender', m[1], train_feature, gender, train_ids)

    print(feat.shape)


#### 用训练好的模型预测，用作stacking特征
train_preds = [pd.DataFrame() for fold in range(5)] 
test_preds = pd.DataFrame()

for col in cols:
    # data_size = [600009, 600001, 600001, 599997, 599992]
    nrows = None
    train_valid_click_log = pd.DataFrame()
    data_size = list()
    for i in range(5):
        df = pd.read_csv(path.format('folds/fold_%d' % (i+1), 'origin_feature_reindexed_fromzero.csv'), usecols=['age', 'gender'], nrows=nrows)
        train_valid_click_log = pd.concat([train_valid_click_log, df], 0)
        data_size.append(df.shape[0])
    age, gender = train_valid_click_log.age.values - 1, train_valid_click_log.gender.values - 1

    train_ids = list()
    pos = 0
    for i in range(len(data_size)):
        train_ids.append(np.arange(data_size[i]) + pos)
        pos += data_size[i]
    print([(min(x), max(x)) for x in train_ids])
    
    discuss_tf = load_npz(path.format('test_preliminary', '%s_tfvec_csr.npz' % col))
    train_feature = discuss_tf[:sum(data_size)]
    test_feature = discuss_tf[sum(data_size):]

    model_list = [
        ['LogisticRegression', ],
        ['SGDClassifier', ],
        ['PassiveAggressiveClassifier', ],
        ['LinearSVC', ]
    ]

    for model in model_list:
        print(model[0])
        # age
        pred_train, pred_test = predict_sklearn_model(col + '_' + model[0], train_feature, age, test_feature, train_ids)
        print([x.shape for x in pred_train], pred_test.shape)

        for fold in range(5):
            train_preds[fold] = pd.concat([train_preds[fold], pred_train[fold]], 1)
        test_preds = pd.concat([test_preds, pred_test], 1)

        print([x.shape for x in train_preds], test_preds.shape)

    for model in model_list:
        print(model[0])
        # gender
        pred_train, pred_test = predict_sklearn_model(col + '_' + model[0] + '_gender', train_feature, gender, test_feature, train_ids)
        print([x.shape for x in pred_train], pred_test.shape)

        for fold in range(5):
            train_preds[fold] = pd.concat([train_preds[fold], pred_train[fold]], 1)
        test_preds = pd.concat([test_preds, pred_test], 1)

        print([x.shape for x in train_preds], test_preds.shape)
# 保存
for fold in range(5):
    train_preds[fold].to_csv(path.format("folds/fold_%d" % (fold + 1), "sklearn_pred_feat.csv"), index=False)
test_preds.to_csv(path.format("test_preliminary", "sklearn_pred_feat.csv"), index=False)
print([x.shape for x in train_preds], test_preds.shape)



    

