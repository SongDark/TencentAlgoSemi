import pandas as pd 
import numpy as np 
import time
import os 


def merge(phase='valid_offline'):
    """
    对训练集/验证集/测试集中的特征按照user_id进行拼接聚合
    :param phase: 训练集/验证集/测试集的文件夹名
    :return:
    """
    def func(x):
        return " ".join(list(x))

    features = ['time', 'creative_id', 'click_times', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']

    path = "../data/precompetition/{phase}/{file}"

    click_path = path.format(phase=phase, file='click_log.csv')
    # ad_path = path.format(phase=phase, file='ad.csv')
    if 'test' in phase:
        ad_path = path.format(phase='test_preliminary', file='ad.csv')
    else:
        ad_path = path.format(phase='train_preliminary', file='ad.csv')
    user_path = path.format(phase=phase, file='user.csv')

    click_log = pd.read_csv(click_path, nrows=None)
    ad_log = pd.read_csv(ad_path)

    partNum = 10
    users = list(set(click_log.user_id))
    users = np.array_split(users, partNum) #将用户分成10组, 只是为了节省内存
    print(len(users), [len(x) for x in users])
        
    if os.path.exists(user_path): #训练集和验证集
        user_log = pd.read_csv(user_path)
        
        res = None 
        for i in range(partNum): #对于每一组用户（随机分的），得到一份part_user、part_click_log（合并了click_log、ad_log和user_log之后的）
            part_user = pd.merge(user_log, pd.DataFrame(users[i], columns=['user_id']), 'inner', 'user_id')
            part_click_log = pd.merge(click_log, part_user[['user_id']], 'inner', 'user_id')

            part_click_log = pd.merge(part_click_log, ad_log, 'left', 'creative_id')
            part_click_log = pd.merge(part_click_log, user_log, 'left', 'user_id')

            part_click_log.sort_values('time', inplace=True)

            for col in features:
                part_click_log[col] = part_click_log[col].astype(str)
                tmp = part_click_log[['user_id', col]].groupby('user_id').aggregate({'user_id': lambda x:list(x)[0], col: func}).reset_index(drop=True) #[0]是什么？
                part_user = pd.merge(part_user, tmp, 'left', 'user_id')
            
            if i == 0:
                res = part_user 
            else:
                res = pd.concat([res, part_user], axis=0)

            print(part_user.shape, res.shape)
        
        res.sort_values("user_id", inplace=True)
        res.to_csv(path.format(phase=phase, file='origin_feature.csv'), index=False)
    
    else: #测试集，没有user_log
        res = None 
        for i in range(partNum):
            part_user = pd.DataFrame(users[i], columns=['user_id'])
            part_click_log = pd.merge(click_log, part_user[['user_id']], 'inner', 'user_id')

            part_click_log = pd.merge(part_click_log, ad_log, 'left', 'creative_id')

            part_click_log.sort_values('time', inplace=True)

            for col in features:
                part_click_log[col] = part_click_log[col].astype(str)
                tmp = part_click_log[['user_id', col]].groupby('user_id').aggregate({'user_id': lambda x:list(x)[0], col: func}).reset_index(drop=True)
                part_user = pd.merge(part_user, tmp, 'left', 'user_id')
            
            if i == 0:
                res = part_user 
            else:
                res = pd.concat([res, part_user], axis=0)

            print(part_user.shape, res.shape)
        
        res.sort_values("user_id", inplace=True)
        res.to_csv(path.format(phase=phase, file='origin_feature.csv'), index=False)

def merge_semi(stage='semi', phase='folds/fold_1'):
    """
    对训练集/验证集/测试集中的特征按照user_id进行拼接聚合
    :param phase: 训练集/验证集/测试集的文件夹名
    :return:
    """

    def func(x):
        return " ".join(list(x))

    features = ['time', 'creative_id', 'click_times', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']

    path = "../data/{stage}/{phase}/{file}"

    click_path = path.format(stage=stage, phase=phase, file='click_log.csv')
    click_log = pd.read_csv(click_path, nrows=None)

    ad_log = pd.concat([
        pd.read_csv(path.format(stage='precompetition', phase='train_preliminary', file='ad.csv')),
        pd.read_csv(path.format(stage='precompetition', phase='test_preliminary', file='ad.csv')),
        pd.read_csv(path.format(stage='semi', phase='train_semi_final', file='ad.csv'))
    ], axis=0).drop_duplicates()

    partNum = 10
    users = list(set(click_log.user_id))
    users = np.array_split(users, partNum) #将用户分成10组, 只是为了节省内存
    print(len(users), [len(x) for x in users])
    
    user_path = path.format(stage=stage, phase=phase, file='user.csv')
    user_log = pd.read_csv(user_path)
    
    res = None 
    for i in range(partNum): #对于每一组用户（随机分的），得到一份part_user、part_click_log（合并了click_log、ad_log和user_log之后的）
        part_user = pd.merge(user_log, pd.DataFrame(users[i], columns=['user_id']), 'inner', 'user_id')
        part_click_log = pd.merge(click_log, part_user[['user_id']], 'inner', 'user_id')

        part_click_log = pd.merge(part_click_log, ad_log, 'left', 'creative_id')
        part_click_log = pd.merge(part_click_log, user_log, 'left', 'user_id')

        part_click_log.sort_values('time', inplace=True)

        for col in features:
            part_click_log[col] = part_click_log[col].astype(str)
            tmp = part_click_log[['user_id', col]].groupby('user_id').aggregate({'user_id': lambda x:list(x)[0], col: func}).reset_index(drop=True) #[0]是什么？
            part_user = pd.merge(part_user, tmp, 'left', 'user_id')
        
        if i == 0:
            res = part_user 
        else:
            res = pd.concat([res, part_user], axis=0)

        print(part_user.shape, res.shape)
    
    res.sort_values("user_id", inplace=True)
    res.to_csv(path.format(stage=stage, phase=phase, file='origin_feature.csv'), index=False)

            
if __name__ == '__main__':

    # 测试集
    merge('test_preliminary')

    # 训练集
    for i in range(5):
        merge_semi(stage='semi', phase='folds/fold_{}'.format(i + 1))



