import pandas as pd 
import numpy as np
import lightgbm as lgb 
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


AGE_PROB_COLS = ['predicted_age_%d' % (i+1) for i in range(10)]
GENDER_PROB_COLS = ['predicted_gender_%d' % (i+1) for i in range(2)]

STACKING_FILES = {
    'cnn': {'valid':['../result/valid_proba_cnn_20200716fold{}.csv'.format(i) for i in np.arange(5) + 1], 
            'test':['../result/proba_cnn_20200716fold{}.csv'.format(i) for i in np.arange(5) + 1]},
    'esim_naive': {'valid':['../result/valid_proba_esim_20200716fold{}.csv'.format(i) for i in np.arange(5) + 1], 
                   'test':['../result/proba_esim_20200716fold{}.csv'.format(i) for i in np.arange(5) + 1]},
    'sklearn_concat_esim': {'valid':['../result/valid_prob_{}_p20200713.csv'.format(i) for i in np.arange(5) + 1], 
                            'test':['../result/test_prob_{}_p20200713.csv'.format(i) for i in np.arange(5) + 1]},
    'transformer': {'valid':['../result/valid_proba_multihead_20200716fold{}.csv'.format(i) for i in np.arange(5) + 1],
                    'test':['../result/proba_multihead_20200716fold{}.csv'.format(i) for i in np.arange(5) + 1]},
}


def make_stacking_input_from_nn(prob_path_dict):

    full_train_feat = pd.DataFrame()
    full_test_feat = pd.DataFrame()
    columns = list(['bizuin',])

    for model in prob_path_dict:
        print(model)
        columns.extend(['%s_%s' % (model, c) for c in AGE_PROB_COLS + GENDER_PROB_COLS])

        # 完整训练集 300w = 5个fold拼接
        full_train_prob = pd.concat([pd.read_csv(f, usecols=['user_id',] + AGE_PROB_COLS + GENDER_PROB_COLS) for f in prob_path_dict[model]['valid']], axis=0)
        full_train_prob.sort_values("user_id", inplace=True)
        full_train_prob.reset_index(inplace=True, drop=True)

        # 测试集 = 5个fold平均
        test_prob = pd.DataFrame()
        for f in prob_path_dict[model]['test']:
            tmp = pd.read_csv(f, usecols=['user_id',] + AGE_PROB_COLS + GENDER_PROB_COLS)
            tmp.sort_values("user_id", inplace=True)
            assert len(tmp) == 1000000
            if test_prob.shape[0] == 0:
                test_prob = tmp
            else:
                test_prob[AGE_PROB_COLS + GENDER_PROB_COLS] += tmp[AGE_PROB_COLS + GENDER_PROB_COLS]
        test_prob[AGE_PROB_COLS + GENDER_PROB_COLS] /= len(prob_path_dict[model]['test'])
        test_prob.reset_index(inplace=True, drop=True)

        print("train_size=%s, test_size=%s" % (str(full_train_prob.shape), str(test_prob.shape)))

        if full_train_feat.shape[0] == 0:
            full_train_feat = full_train_prob
            full_test_feat = test_prob
        else:
            full_train_feat = pd.concat([full_train_feat, full_train_prob[AGE_PROB_COLS + GENDER_PROB_COLS]], axis=1)
            full_test_feat = pd.concat([full_test_feat, test_prob[AGE_PROB_COLS + GENDER_PROB_COLS]], axis=1)
    
        print("train_size=%s, test_size=%s" % (str(full_train_feat.shape), str(full_test_feat.shape)))
    
    full_train_feat.columns = columns
    full_test_feat.columns = columns

    full_train_feat.to_csv("../data/semi/stacking/nn_predicted_prob_train.csv", index=False)
    full_test_feat.to_csv("../data/semi/stacking/nn_predicted_prob_test.csv", index=False)



age_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_error',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': 0,
    'feature_fraction': 0.6,
    'lambda_l1': 0.01,
    "num_class":10,
}
gender_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': 0,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.6,
    'lambda_l1': 0.01,
}



def compute_metrics_multi(label, prob):
    pred = np.argmax(prob, 1)
    res = [accuracy_score(label, pred)]
    res = [np.round(x, 5) for x in res]
    return res


def lgb_train():
    weighted = ''

    train_valid_feat = pd.read_csv("../data/semi/stacking/{}nn_predicted_prob_train.csv".format(weighted) )
    train_valid_label = pd.concat([pd.read_csv("../data/semi/folds/fold_{}/origin_feature_reindexed_fromzero.csv".format(fold), usecols=['user_id', 'age', 'gender'] ) for fold in np.arange(5) + 1], 0)
    train_valid_label.sort_values("user_id", inplace=True)
    assert len(train_valid_feat) == len(train_valid_label)

    # split and train: age
    skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for fold_id, (train_index, valid_index) in enumerate(skf.split(train_valid_feat, train_valid_label.age.values)):
        train_feat = train_valid_feat.values[train_index, 1:]
        valid_feat = train_valid_feat.values[valid_index, 1:]
        train_label = train_valid_label['age'].values[train_index] - 1
        valid_label = train_valid_label['age'].values[valid_index] - 1

        model = lgb.train(age_params, lgb.Dataset(train_feat, train_label), num_boost_round=2000,
                          valid_sets=lgb.Dataset(valid_feat, valid_label), early_stopping_rounds=10)
        model.save_model("../models/lgbm/model_{}age_fold{}.txt".format(weighted, fold_id))
        train_prob = model.predict(train_feat)
        valid_prob = model.predict(valid_feat)
        print("train:", compute_metrics_multi(train_label, train_prob))
        print("valid:", compute_metrics_multi(valid_label, valid_prob))
    
    # split and train: age
    skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for fold_id, (train_index, valid_index) in enumerate(skf.split(train_valid_feat, train_valid_label.gender.values)):
        train_feat = train_valid_feat.values[train_index, 1:]
        valid_feat = train_valid_feat.values[valid_index, 1:]
        train_label = train_valid_label['gender'].values[train_index] - 1
        valid_label = train_valid_label['gender'].values[valid_index] - 1

        model = lgb.train(gender_params, lgb.Dataset(train_feat, train_label), num_boost_round=2000,
                          valid_sets=lgb.Dataset(valid_feat, valid_label), early_stopping_rounds=10)
        model.save_model("../models/lgbm/model_{}gender_fold{}.txt".format(weighted, fold_id))
        model.predict_proba = lambda x: np.vstack([1 - model.predict(x), model.predict(x)]).T
        train_prob = model.predict_proba(train_feat)
        valid_prob = model.predict_proba(valid_feat)
        print("train:", compute_metrics_multi(train_label, train_prob))
        print("valid:", compute_metrics_multi(valid_label, valid_prob))
        
def lgb_predict():
    weighted = ''

    train_valid_feat = pd.read_csv("../data/semi/stacking/{}nn_predicted_prob_train.csv".format(weighted ))
    train_valid_label = pd.concat([pd.read_csv("../data/semi/folds/fold_{}/origin_feature_reindexed_fromzero.csv".format(fold), usecols=['user_id', 'age', 'gender']) for fold in np.arange(5) + 1], 0)
    train_valid_label.sort_values("user_id", inplace=True)
    test_feat = pd.read_csv("../data/semi/stacking/{}nn_predicted_prob_test.csv".format(weighted))
    test_feat.columns = ['user_id'] + list(test_feat.columns)[1:]
    assert len(train_valid_feat) == len(train_valid_label)

    test_prob = test_feat[['user_id']]
    for col in AGE_PROB_COLS + GENDER_PROB_COLS:
        test_prob[col] = np.zeros((len(test_feat), 1))

    # split and train: age
    skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for fold_id, (train_index, valid_index) in enumerate(skf.split(train_valid_feat, train_valid_label.age.values)):
        valid_feat = train_valid_feat.values[valid_index, 1:]
        valid_label = train_valid_label['age'].values[valid_index] - 1

        model = lgb.Booster(model_file="../models/lgbm/model_{}age_fold{}.txt".format(weighted, fold_id))
        valid_prob = model.predict(valid_feat)
        print("valid age", fold_id, compute_metrics_multi(valid_label, valid_prob))

        test_prob[AGE_PROB_COLS] += model.predict(test_feat.values[:, 1:])

        if fold_id == 0:
            tmp = pd.DataFrame({"pred": np.argmax(valid_prob, 1) + 1, "tmp": 0}).groupby("pred").size().reset_index()
            tmp[0] /= float(len(valid_feat))
            print(tmp)

    
    # split and train: age
    skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for fold_id, (train_index, valid_index) in enumerate(skf.split(train_valid_feat, train_valid_label.gender.values)):
        valid_feat = train_valid_feat.values[valid_index, 1:]
        valid_label = train_valid_label['gender'].values[valid_index] - 1

        model = lgb.Booster(model_file="../models/lgbm/model_{}gender_fold{}.txt".format(weighted, fold_id))
        model.predict_proba = lambda x: np.vstack([1 - model.predict(x), model.predict(x)]).T
        valid_prob = model.predict_proba(valid_feat)
        print("valid gender", fold_id, compute_metrics_multi(valid_label, valid_prob))

        test_prob[GENDER_PROB_COLS] += model.predict_proba(test_feat.values[:, 1:])

        if fold_id == 0:
            tmp = pd.DataFrame({"pred": np.argmax(valid_prob, 1) + 1, "tmp": 0}).groupby("pred").size().reset_index()
            tmp[0] /= float(len(valid_feat))
            print(tmp)
    
    test_prob['predicted_age'] = np.argmax(test_prob[AGE_PROB_COLS].values, 1) + 1
    test_prob['predicted_gender'] = np.argmax(test_prob[GENDER_PROB_COLS].values, 1) + 1

    tmp = test_prob[['predicted_age', 'user_id']].groupby("predicted_age").size().reset_index()
    tmp[0] /= len(test_feat) 
    print(tmp)
    tmp = test_prob[['predicted_gender', 'user_id']].groupby("predicted_gender").size().reset_index()
    tmp[0] /= len(test_feat)
    print(tmp)

    test_prob[['user_id', 'predicted_age', 'predicted_gender']].to_csv("../submission.csv", index=False)
        


if __name__ == '__main__':
    make_stacking_input_from_nn(STACKING_FILES)

    lgb_train()

    lgb_predict()