echo "begin"
cd src
echo "data process"
python data_divide.py 
python data_merge.py
python feature_count.py
python data_reindexed.py 
echo "linear stacking"
python stacking_from_linear.py
echo "train esim conat" 
python esim_concat.py train_kfold_k 1 final
python esim_concat.py train_kfold_k 2 final
python esim_concat.py train_kfold_k 3 final
python esim_concat.py train_kfold_k 4 final
python esim_concat.py train_kfold_k 5 final
echo "stacking"
python stacking_from_nn.py 