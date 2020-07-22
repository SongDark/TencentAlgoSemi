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
echo "train esim"
python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024
python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024
python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024
python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024
python model_esim.py --MODE train_kfold_k --VERSION esim_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024
python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024
python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024
python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024
python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024
python model_esim.py --MODE pred_kfold_k --VERSION esim_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024
echo "train cnn"
python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
python model_esim_cnn.py --MODE train_kfold_k --VERSION cnn_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 1 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 2 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 3 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 4 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
python model_esim_cnn.py --MODE pred_kfold_k --VERSION cnn_20200716 --FOLDK 5 --NUM_GPU 1 --BATCH_SIZE 1024 --use_CuDNNLSTM True
echo "stacking and make submission"
python stacking_from_nn.py 