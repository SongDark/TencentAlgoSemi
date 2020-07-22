echo "begin"
cd src
ls
echo "data process"
python data_divide.py 
python data_merge.py
python feature_count.py
python data_reindexed.py 
echo "linear stacking"
python stacking_from_linear.py
echo "train esim conat" 
python esim_concat.py train_kfold_k 1 p20200713
python esim_concat.py train_kfold_k 2 p20200713
python esim_concat.py train_kfold_k 3 p20200713
python esim_concat.py train_kfold_k 4 p20200713
python esim_concat.py train_kfold_k 5 p20200713
python esim_concat.py predict_kfold_k 1 p20200713
python esim_concat.py predict_kfold_k 2 p20200713
python esim_concat.py predict_kfold_k 3 p20200713
python esim_concat.py predict_kfold_k 4 p20200713
python esim_concat.py predict_kfold_k 5 p20200713
echo "train nn"
echo "stacking"
python stacking_from_nn.py 