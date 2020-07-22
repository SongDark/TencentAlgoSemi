echo "begin"
cd src
echo "data process"
python data_divide.py 
python data_merge.py
python feature_count.py
python data_reindexed.py 
echo "linear stacking"
python stacking_from_linear.py 
python stacking_from_nn.py 