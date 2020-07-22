#encoding:utf-8
import pandas as pd 
import numpy as np 

def divide_into_folds_semi(fold=5):
    # 保证每个fold中，age-gender的分布与大盘一致
    N = None
    train_user_log = pd.concat(
        [
            pd.read_csv('../data/precompetition/train_preliminary/user.csv', nrows=N),
            pd.read_csv('../data/semi/train_semi_final/user.csv', nrows=N)
        ], 
        axis=0
    ).sort_values('user_id').reset_index()
    train_user_log['fold'] = -1
    
    train_user_log['label'] = train_user_log.apply(lambda x: str(x['age']) + '-' + str(x['gender']), axis=1)

    grouped = train_user_log[['label', 'user_id']].groupby('label').count().reset_index()  # 20类
    # print(grouped)
    
    labels = sorted(list(set(grouped['label'].values)))

    fold_index_dict = {i : list() for i in range(fold)}
    for i in range(len(labels)): #对于每种age-gender
        cur_label = labels[i]

        current_label_user_index = list(train_user_log[train_user_log['label'] == cur_label].index) #取出该记录的index

        np.random.seed(i)
        np.random.shuffle(current_label_user_index)
        current_label_user_index = np.array_split(current_label_user_index, fold) #将index分成5组

        for j in range(fold):
            fold_index_dict[j].extend(current_label_user_index[j]) #将index放到各组的list中

    # print([(k, len(v)) for k,v in fold_index_dict.items()])

    for i in range(fold):
        train_user_log.loc[fold_index_dict[i], 'fold'] = i + 1 #根据各组的index，设置其'fold'列的值，值为1～5
        
    # train_user_log.to_csv("../data/semi/train_preliminary/user_divided.csv", index=False)

    # train_user_log = pd.read_csv("../data/semi/train_preliminary/user_divided.csv")
    # print(train_user_log[['user_id', 'fold']].groupby('fold').count().reset_index())

    train_click_log = pd.concat(
        [
            pd.read_csv("../data/precompetition/train_preliminary/click_log.csv"),
            pd.read_csv("../data/semi/train_semi_final/click_log.csv")
        ],
        axis=0
    )
    for i in range(fold):
        pd.merge(train_click_log, train_user_log[train_user_log.fold == i + 1]['user_id'], 'inner', 'user_id').drop_duplicates().to_csv("../data/semi/folds/fold_{}/click_log.csv".format(i+1), index=False)
        train_user_log[train_user_log.fold == i + 1][['user_id', 'age', 'gender']].drop_duplicates().to_csv("../data/semi/folds/fold_{}/user.csv".format(i+1), index=False)


if __name__ == '__main__':
    divide_into_folds_semi(5)
    
