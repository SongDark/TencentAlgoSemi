# 腾讯算法广告大赛2020

# 概述
腾讯算法广告大赛2020代码

# 环境

- numpy==1.18.4
- pandas==0.25.2
- scipy==1.4.1
- tensorflow==1.12
- keras==2.2.4
- gensim==3.8.1
- sklearn==0.22.2.post1
- lightgbm==2.3.1

# 代码/数据结构

## 数据准备
把解压后初赛的数据放在`data/precompetition/`下，复赛的数据放在`data/semi/`下。

```
src # 代码
data # 数据
|- precompetition # 初赛数据
|-- train_preliminary
|--- ad.csv
|--- click_log.csv
|--- user.csv
|-- test_preliminary
|--- ad.csv
|--- click_log.csv
|- semi # 复赛数据
|-- train_semi_final
|--- ad.csv
|--- click_log.csv
|--- user.csv
|-- test_preliminary # 同初赛
|--- origin_feature_reindexed_fromzero.csv
|--- sklearn_pred_feat.csv
|-- dict # 字典数据
|-- folds # 训练集
|--- fold_1
|---- origin_feature_reindexed_fromzero.csv
|---- sklearn_pred_feat.csv
|--- fold_2
|--- fold_3
|--- fold_4
|--- fold_5
|-- stacking # stacking训练数据
|-- mid # 中间结果
models # 模型
|- esim_concat
|- sklearn_linear
|- lgbm
|- tfidf
|- w2v
result # 中间结果
submission.csv  # 最终提交
run.sh  # 运行入口
``` 

# 运行入口

```shell
$   ./run.sh
```

# 详细逻辑

## 1. 数据预处理

### 数据切分
在 `data/semi/folds/fold_*/` 下产生 `user.csv`、`click_log.csv`
```shell
$   python data_divide.py
```

### 原始特征合并
在 `data/semi/folds/fold_*/` 下产生 `origin_feature.csv`
```shell
$   python data_merge.py
```

### 计算词频
在 `data/semi/dict/` 下产生 `feature_cnt.pkl`
```shell
$   python feature_count.py
```

### 重映射
在 `data/semi/folds/fold_*/` 下产生 `origin_feature_reindexed_fromzero.csv`
```shell
$   python data_reindexed.py
```

### 线性模型（预测值用于Stacking）
在 `models/tfidf/` 下产生 `*_tfvec_csr.npz`
在 `data/semi/folds/fold_*/` 下产生 `sklearn_pred_feat.csv`
```shell
$   python stacking_from_linear.py
```

## 2. 预训练word2vec词向量
在 `models/w2v/` 下产生 `w2v_embeddings_*_p20200628.npy`
```shell
$   python word2vec.py
```

## 3. 模型训练
在 `models/` 下产生 `model_{version}_fold{k}.h`
```shell
$   python esim_concat.py train_kfold_k 1 p20200713
$   python esim_concat.py train_kfold_k 2 p20200713
$   python esim_concat.py train_kfold_k 3 p20200713
$   python esim_concat.py train_kfold_k 4 p20200713
$   python esim_concat.py train_kfold_k 5 p20200713

$   python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024
$   python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024
$   python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024
$   python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024
$   python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024

$   python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True

$   python model_multihead_cnn.py --MODE train_kfold_k --VERSION multihead_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_multihead_cnn.py --MODE train_kfold_k --VERSION multihead_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_multihead_cnn.py --MODE train_kfold_k --VERSION multihead_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_multihead_cnn.py --MODE train_kfold_k --VERSION multihead_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_multihead_cnn.py --MODE train_kfold_k --VERSION multihead_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
```

## 4. 模型预测
在 `result/` 下产生测试集的预测概率 `proba_{version}_fold{k}.csv` 和验证集的预测概率 `valid_proba_{version}_fold{k}.csv`
```shell
$   python esim_concat.py predict_kfold_k 1 p20200713
$   python esim_concat.py predict_kfold_k 2 p20200713
$   python esim_concat.py predict_kfold_k 3 p20200713
$   python esim_concat.py predict_kfold_k 4 p20200713
$   python esim_concat.py predict_kfold_k 5 p20200713

$   python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024
$   python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024
$   python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024
$   python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024
$   python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024

$   python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True

$   python model_multihead_cnn.py --MODE pred_kfold_k --VERSION multihead_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_multihead_cnn.py --MODE pred_kfold_k --VERSION multihead_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_multihead_cnn.py --MODE pred_kfold_k --VERSION multihead_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_multihead_cnn.py --MODE pred_kfold_k --VERSION multihead_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
$   python model_multihead_cnn.py --MODE pred_kfold_k --VERSION multihead_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
```

## 5. Stacking 并产生最终提交结果
```shell
$   python stacking_from_nn.py
```