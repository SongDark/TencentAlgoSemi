# Readme

# 数据结构

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
|-- dict # 字典数据
|--- feature_cnt.pkl
|-- folds # 训练集
|--- fold_1
|---- origin_feature_reindexed_fromzero.csv
|--- fold_2
|--- fold_3
|--- fold_4
|--- fold_5
models # 模型
``` 

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

## 2. 线性模型（预测值用于Stacking）
在 `data/semi/folds/fold_*/` 下产生 `sklearn_pred_feat.csv`
```shell
$   python stacking_from_linear.py
```