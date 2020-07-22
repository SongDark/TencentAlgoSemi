#coding:utf-8

"""安装和导入需要的包"""
from pip._internal import main
main(['install', "keras==2.2.4"])
# import keras
# print(keras.__version__)
from keras.layers import Input, Dense, Embedding, Concatenate, Dropout, MaxPooling1D, Masking, Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import pickle
import h5py
# from keras_transformer import get_encoder_component, get_encoders
# from keras_transformer.gelu import gelu
import sys
from keras.layers import *
from keras.models import Model, load_model
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.activations import softmax
from keras.utils import multi_gpu_model
from keras import regularizers

"""设置输入路径、输出路径、参数"""
import argparse
parser = argparse.ArgumentParser()
# 不怎么需要改的
parser.add_argument('--LATENT_DIM', default=200, type=int,
                   help='word2vec size')
parser.add_argument('--MAX_SEQ_LEN', default=175, type=int,
                   help='number of training steps')
parser.add_argument('--TRAIN_EPOCH', default=20, type=int,
                   help='number of training steps')
parser.add_argument('--W2V_PATH', default='../models/w2v/', type=str,
                   help='dir path to the w2v embedding file') #'${cos}/semifinal/w2v/'
parser.add_argument('--W2V_FILE_NAME', default='w2v_embeddings_{col}_p20200628.npy', type=str,
                   help='file name of the w2v embedding file')
parser.add_argument('--TRAIN_FOLD1_PATH', default='../data/semi/folds/fold_1/', type=str,
                   help='dir path to the train data fold1 file')  #'${cos}/semifinal/data/fold_1/'
parser.add_argument('--TRAIN_FOLD2_PATH', default='../data/semi/folds/fold_2/', type=str,
                   help='dir path to the train data fold2 file')  #'${cos}/semifinal/data/fold_2/'
parser.add_argument('--TRAIN_FOLD3_PATH', default='../data/semi/folds/fold_3/', type=str,
                   help='dir path to the train data fold3 file')  #'${cos}/semifinal/data/fold_3/'
parser.add_argument('--TRAIN_FOLD4_PATH', default='../data/semi/folds/fold_4/', type=str,
                   help='dir path to the train data fold4 file')  #'${cos}/semifinal/data/fold_4/'
parser.add_argument('--TRAIN_FOLD5_PATH', default='../data/semi/folds/fold_5/', type=str,
                   help='dir path to the train data fold5 file')  #'${cos}/semifinal/data/fold_5/'
parser.add_argument('--TEST_PATH', default='../data/precompetition/test_preliminary/', type=str,
                   help='dir path to the test data file')  #'${cos}/semifinal/data/test_preliminary/'
parser.add_argument('--DATA_FILE_NAME', default='origin_feature_reindexed_fromzero.csv', type=str,
                   help='file name of the data file')
parser.add_argument('--MODEL_PATH', default='../models/', type=str,
                   help='dir path to export the train model') #'${cos}/semifinal/model/'
parser.add_argument('--MODEL_FILE_NAME', default='model_{version}.h', type=str,
                    help='file name of model')
parser.add_argument('--MODEL_KFOLD_FILE_NAME', default='model_{version}_fold{k}.h', type=str,
                    help='file name of model foldk')
parser.add_argument('--SUBMISSION_PATH', default='../result/',  type=str,
                   help='dir path to export the result path') #'${cos}/semifinal/result/'

# 中间的输出
parser.add_argument('--MID_DATA_PATH', default='../data/semi/mid/', type=str,
                    help="dir path to mid data") #'${cos}/semifinal/data/mid/'
parser.add_argument('--MID_DATA_TRAIN_ALL_FILE_NAME', default='data_train_all.csv', type=str,
                    help="file name of train data all")
parser.add_argument('--MID_DATA_TRAIN_FOLDK_FILE_NAME', default='data_train_fold{k}.csv', type=str,
                    help="file name of train data foldk")
parser.add_argument('--MID_DATA_VALID_FOLDK_FILE_NAME', default='data_valid_fold{k}.csv', type=str,
                    help="file name of valid data foldk")

# 需要改的
parser.add_argument('--BATCH_SIZE', default=1024, type=int,
                   help='batch size')
parser.add_argument('--NUM_GPU', default=1, type=int,
                   help='num of used gpu')
parser.add_argument('--MODE', default='train_kfold_k', type=str,
                    help='train_kfold_k, continue_train_model_kfold_k, pred_kfold_k')
parser.add_argument('--VERSION', default='test_20200702', type=str,
                    help='output file version postfix')
parser.add_argument('--FOLDK', default=1, type=int,
                    help='1,2,3,4,5')
parser.add_argument('--use_CuDNNLSTM', default=False, type=bool,
                    help="use CuDNNLSTM or not")

def main(argv):
    args = parser.parse_args(argv[1:])
    LATENT_DIM = args.LATENT_DIM
    MAX_SEQ_LEN = args.MAX_SEQ_LEN
    BATCH_SIZE = args.BATCH_SIZE
    TRAIN_EPOCH = args.TRAIN_EPOCH
    W2V_PATH = args.W2V_PATH
    W2V_FILE_NAME = args.W2V_FILE_NAME
    TRAIN_FOLD1_PATH = args.TRAIN_FOLD1_PATH
    TRAIN_FOLD2_PATH = args.TRAIN_FOLD2_PATH
    TRAIN_FOLD3_PATH = args.TRAIN_FOLD3_PATH
    TRAIN_FOLD4_PATH = args.TRAIN_FOLD4_PATH
    TRAIN_FOLD5_PATH = args.TRAIN_FOLD5_PATH
    TEST_PATH = args.TEST_PATH
    DATA_FILE_NAME = args.DATA_FILE_NAME
    MODEL_PATH = args.MODEL_PATH
    SUBMISSION_PATH = args.SUBMISSION_PATH
    MODE = args.MODE
    VERSION = args.VERSION
    FOLDK = args.FOLDK
    MID_DATA_PATH = args.MID_DATA_PATH
    MID_DATA_TRAIN_ALL_FILE_NAME = args.MID_DATA_TRAIN_ALL_FILE_NAME
    MID_DATA_TRAIN_FOLDK_FILE_NAME = args.MID_DATA_TRAIN_FOLDK_FILE_NAME
    MID_DATA_VALID_FOLDK_FILE_NAME = args.MID_DATA_VALID_FOLDK_FILE_NAME
    MODEL_FILE_NAME = args.MODEL_FILE_NAME
    MODEL_KFOLD_FILE_NAME = args.MODEL_KFOLD_FILE_NAME
    use_CuDNNLSTM = args.use_CuDNNLSTM
    NUM_GPU = args.NUM_GPU

    MODEL_FILE_NAME = MODEL_FILE_NAME.format(version=VERSION)
    print(MODEL_FILE_NAME)

    class MyMaxPool(Layer):
        def __init__(self, axis=1, **kwargs):
            self.supports_masking = True
            self.axis = axis
            super(MyMaxPool, self).__init__(**kwargs)

        def compute_mask(self, input, input_mask=None):
            # need not to pass the mask to next layers
            return None

        def call(self, x, mask=None):
            if mask is not None:
                if K.ndim(x)!=K.ndim(mask):
                    mask = K.repeat(mask, x.shape[-1])
                    mask = tf.transpose(mask, [0,2,1])
                mask = K.cast(mask, K.floatx())
                x = x * mask
                return K.max(x, axis=self.axis, keepdims=False)
            else:
                return K.max(x, axis=self.axis, keepdims=False)

        def compute_output_shape(self, input_shape):
            output_shape = []
            for i in range(len(input_shape)):
                if i!=self.axis:
                    output_shape.append(input_shape[i])
            return tuple(output_shape)


    def one_hot_encode(ys, max_class):
        res = np.zeros((len(ys), max_class), dtype=np.float32)
        for i in range(len(ys)):
            res[i][ys[i]] = 1.0
        return res

    def padding(minibatch, maxlen=None):
        lens = [len(vec) for vec in minibatch]
        dim = list(minibatch[0].shape[1:])
        maxlen = maxlen or max(lens)
        res = []
        for i in range(len(minibatch)):
            # print i
            if len(minibatch[i]) > maxlen:
                res.append(minibatch[i][:maxlen, :])
            else:
                res.append(np.concatenate([minibatch[i], np.zeros([maxlen-lens[i]]+dim, dtype=np.int32)], axis=0))
        return np.asarray(res)

    class MultiHeadAttention(Layer):
        def __init__(self, d_model, num_heads):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model

            # d_model 必须可以正确分为各个头
            assert d_model % num_heads == 0

            # 分头后的维度
            self.depth = d_model // num_heads

        # def build(self, input_shape):
        #     # print(input_shape) # [(None, 175, 800), (None, 175, 800), (None, 175, 800)]
        #     self._weights_queries = self.add_weight(
        #         shape=(int(input_shape[0][-1]), self.num_heads * self.depth),
        #         initializer='glorot_uniform',
        #         trainable=True,
        #         name='weights_queries'
        #     )
        #     self._weights_keys = self.add_weight(
        #         shape=(int(input_shape[0][-1]), self.num_heads * self.depth),
        #         initializer='glorot_uniform',
        #         trainable=True,
        #         name='weights_keys'
        #     )
        #     self._weights_values = self.add_weight(
        #         shape=(int(input_shape[0][-1]), self.num_heads * self.depth),
        #         initializer='glorot_uniform',
        #         trainable=True,
        #         name='weights_values'
        #     )
        #     self._weights_output = self.add_weight(
        #         shape=(int(input_shape[0][-1]), self.num_heads * self.depth),
        #         initializer='glorot_uniform',
        #         trainable=True,
        #         name='weights_output'
        #     )
        #     super(MultiHeadAttention, self).build(input_shape)

        def split_heads(self, x, batch_size):
            # 分头, 将头个数的维度 放到 seq_len 前面
            x = K.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, inputs):
            if len(inputs)>3:
                q, k, v, mask = inputs
            else:
                q, k, v = inputs
                mask = 0.0
            batch_size = tf.shape(q)[0]

            # # 分头前的前向网络，获取q、k、v语义
            # q = K.dot(q, self._weights_queries)  # (batch_size, seq_len, d_model)
            # k = K.dot(k, self._weights_keys)
            # v = K.dot(v, self._weights_values)
            # # print(q) #shape=(?, 175, 200)

            # 分头
            q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
            k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
            # print(q) #shape=(?, 2, ?, 100)

            # 通过缩放点积注意力层
            scaled_attention = scaled_dot_product_attention(q, k, v, mask)  # (batch_size, num_heads, seq_len_q, depth)
            # print(q) #shape=(?, 2, ?, 100)

            # “多头维度” 后移
            scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
            # print(scaled_attention) #shape=(?, ?, 2, 100)

            # 合并 “多头维度”
            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
            # print(concat_attention) #shape=(?, ?, 200)
            output = concat_attention

            # # 全连接层
            # output = K.dot(concat_attention, self._weights_output)
            return output

    def scaled_dot_product_attention(q, k, v, mask=0.0):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dim_k)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output



    used_cols = ['creative_id', 'advertiser_id', 'ad_id', 'product_id', 'industry']

    # FEAT_CNT = {'creative_id': 2086586, 'time': 91, 'product_id': 31581, 'product_category': 18, 'advertiser_id': 50544, 'industry': 324, 'ad_id': 1977177}
    FEAT_CNT = {'creative_id': 3009975, 'time': 91, 'product_id': 37188, 'product_category': 18, 'advertiser_id': 56354,
                'industry': 334, 'ad_id': 2768715}
    full_cols_dict = {'user_id': 0, 'age': 1, 'gender': 2, 'time': 3, 'creative_id': 4, 'click_times': 5, 'ad_id': 6, 'product_id': 7, 'product_category': 8, 'advertiser_id': 9, 'industry': 10}
    full_cols_dict_test = {'user_id': 0, 'time': 1, 'creative_id': 2, 'click_times': 3, 'ad_id': 4, 'product_id': 5, 'product_category': 6, 'advertiser_id': 7, 'industry': 8}


    W2V_VECTOR = dict()
    for col in used_cols:
        W2V_VECTOR[col] = np.load(os.path.join(W2V_PATH, W2V_FILE_NAME.format(col=col)))


    def process_line(line, stage='train'):
        # global used_cols, W2V_VECTOR

        if stage == 'train':
            vecs = line.split(",")
            age, gender = vecs[full_cols_dict['age']], vecs[full_cols_dict['gender']]
            X = list()
            for col in used_cols:
                col_vec = vecs[full_cols_dict[col]].rstrip().split(' ')[:MAX_SEQ_LEN]
                idxs = [int(k) for k in col_vec]
                X.append(W2V_VECTOR[col][idxs])

            Y = [int(age) - 1, int(gender) - 1]
            return X, Y

        elif stage == 'test':
            vecs = line.split(",")
            X = list()
            for col in used_cols:
                col_vec = vecs[full_cols_dict_test[col]].rstrip().split(' ')
                idxs = [int(k) for k in col_vec]
                X.append(W2V_VECTOR[col][idxs])

            Y = [-1, -1]
            return X, Y


    def generate_from_file(file_path, batch_size=64, stage='train'):
        X, Y = [[] for _ in range(len(used_cols))], [[], []]
        while True:
            with open(file_path, 'r') as fin:
                fin.readline()  # header
                while True:
                    line = fin.readline()
                    if not line:
                        break
                    xs, ys = process_line(line.rstrip(), stage)
                    if xs is None or ys is None:
                        continue
                    for i in range(len(xs)):
                        X[i].append(xs[i])
                    for i in range(len(ys)):
                        Y[i].append(ys[i])

                    if len(X[0]) >= batch_size:
                        yield [padding(x, MAX_SEQ_LEN) for x in X], [one_hot_encode(Y[0], 10), one_hot_encode(Y[1], 2)]
                        X, Y = [[] for _ in range(len(used_cols))], [[], []]



    def build_model():

        input_tensor1 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input1')
        input_tensor2 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input2')
        input_tensor3 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input3')
        input_tensor4 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input4')
        input_tensor5 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input5')

        # masked_input1 = Masking()(input_tensor1)
        # masked_input2 = Masking()(input_tensor2)
        # masked_input3 = Masking()(input_tensor3)
        # masked_input4 = Masking()(input_tensor4)
        # masked_input5 = Masking()(input_tensor5)

        # embedding_sequence_q1 = BatchNormalization(axis=2)(masked_input1)
        # embedding_sequence_q2 = BatchNormalization(axis=2)(masked_input2)
        # embedding_sequence_q3 = BatchNormalization(axis=2)(masked_input3)
        # embedding_sequence_q4 = BatchNormalization(axis=2)(masked_input4)
        # embedding_sequence_q5 = BatchNormalization(axis=2)(masked_input5)

        embedding_sequence_q1 = BatchNormalization(axis=2)(input_tensor1)
        embedding_sequence_q2 = BatchNormalization(axis=2)(input_tensor2)
        embedding_sequence_q3 = BatchNormalization(axis=2)(input_tensor3)
        embedding_sequence_q4 = BatchNormalization(axis=2)(input_tensor4)
        embedding_sequence_q5 = BatchNormalization(axis=2)(input_tensor5)

        final_embedding_sequence_q1 = SpatialDropout1D(0.25)(embedding_sequence_q1)
        final_embedding_sequence_q2 = SpatialDropout1D(0.25)(embedding_sequence_q2)
        final_embedding_sequence_q3 = SpatialDropout1D(0.25)(embedding_sequence_q3)
        final_embedding_sequence_q4 = SpatialDropout1D(0.25)(embedding_sequence_q4)
        final_embedding_sequence_q5 = SpatialDropout1D(0.25)(embedding_sequence_q5)

        final_embedding = concatenate([final_embedding_sequence_q1, final_embedding_sequence_q2, final_embedding_sequence_q3, final_embedding_sequence_q4, final_embedding_sequence_q5])

        # 先理一下时序关系
        if use_CuDNNLSTM:
            rnn_layer_q1 = CuDNNLSTM(LATENT_DIM, return_sequences=True)(final_embedding)
            rnn_layer_q2 = CuDNNLSTM(LATENT_DIM, return_sequences=True)(final_embedding)
        else:
            rnn_layer_q1 = LSTM(LATENT_DIM, return_sequences=True)(final_embedding)
            rnn_layer_q2 = LSTM(LATENT_DIM, return_sequences=True)(final_embedding)

        # # 1. CNN:自身里找最大
        cnn_layer_q1 = Conv1D(filters=LATENT_DIM, kernel_size=1, padding='same', activation='relu')(rnn_layer_q1)
        cnn_layer_q2 = Conv1D(filters=LATENT_DIM, kernel_size=1, padding='same', activation='relu')(rnn_layer_q1)
        cnn_layer_q1 = GlobalMaxPooling1D()(cnn_layer_q1)
        cnn_layer_q2 = GlobalMaxPooling1D()(cnn_layer_q2)

        # 3，Multihead：seq内各时间步间的特征交叉（从头开始比较好）
        attention = Dot(axes=-1)([rnn_layer_q1, rnn_layer_q2])
        w_attn_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
        w_attn_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
        align_layer_1 = Dot(axes=1)([w_attn_1, rnn_layer_q1])
        align_layer_2 = Dot(axes=1)([w_attn_2, rnn_layer_q2])

        subtract_layer_1 = subtract([rnn_layer_q1, align_layer_1])
        subtract_layer_2 = subtract([rnn_layer_q2, align_layer_2])

        multiply_layer_1 = multiply([rnn_layer_q1, align_layer_1])
        multiply_layer_2 = multiply([rnn_layer_q2, align_layer_2])

        m_q1 = concatenate([rnn_layer_q1, align_layer_1, subtract_layer_1, multiply_layer_1])
        m_q2 = concatenate([rnn_layer_q2, align_layer_2, subtract_layer_2, multiply_layer_2])

        v_q1_i = MultiHeadAttention(d_model=800, num_heads=2)([m_q1, m_q1, m_q1])
        v_q2_i = MultiHeadAttention(d_model=800, num_heads=2)([m_q2, m_q2, m_q2])

        avgpool_q1 = GlobalAveragePooling1D()(v_q1_i)
        avgpool_q2 = GlobalAveragePooling1D()(v_q2_i)
        # maxpool_q1 = MyMaxPool(axis=1, name='maxpool_1')(v_q1_i)
        # maxpool_q2 = MyMaxPool(axis=1, name='maxpool_2')(v_q2_i)
        maxpool_q1 = GlobalMaxPooling1D()(v_q1_i)
        maxpool_q2 = GlobalMaxPooling1D()(v_q2_i)
        merged_q1 = concatenate([avgpool_q1, maxpool_q1])
        merged_q2 = concatenate([avgpool_q2, maxpool_q2])

        final_v = BatchNormalization()(concatenate([merged_q1, merged_q2, cnn_layer_q1, cnn_layer_q2])) #改成dropout试试
        print(final_v)

        output_age = Dense(units=64, activation='relu')(final_v)
        output_age = BatchNormalization()(output_age)
        output_age = Dropout(0.3)(output_age)
        y_age = Dense(10, activation='softmax', name='out_age')(output_age)

        output_gender = Dense(units=64, activation='relu')(final_v)
        output_gender = BatchNormalization()(output_gender)
        output_gender = Dropout(0.3)(output_gender)
        y_gender = Dense(2, activation='softmax', name='out_gender')(output_gender)

        model = Model(inputs=[input_tensor1, input_tensor2, input_tensor3, input_tensor4, input_tensor5], outputs=[y_age, y_gender])

        return model

    def calculate_steps(num_samples, num_epoch):
        steps = num_samples // num_epoch
        if num_samples % num_epoch != 0:
            steps += 1
        return steps


    def train_model_kfold_k(k):
        """
        print_lines(train_path) #(719992, 11)
        print_lines(valid_path) #(180008, 11)
        print_lines(test_path) #(1000000, 9)
        :return:
        """

        from sklearn.model_selection import StratifiedKFold

        # path setting
        train_fold1_path = os.path.join(TRAIN_FOLD1_PATH, DATA_FILE_NAME)
        train_fold2_path = os.path.join(TRAIN_FOLD2_PATH, DATA_FILE_NAME)
        train_fold3_path = os.path.join(TRAIN_FOLD3_PATH, DATA_FILE_NAME)
        train_fold4_path = os.path.join(TRAIN_FOLD4_PATH, DATA_FILE_NAME)
        train_fold5_path = os.path.join(TRAIN_FOLD5_PATH, DATA_FILE_NAME)
        train_path = os.path.join(MID_DATA_PATH, MID_DATA_TRAIN_ALL_FILE_NAME)
        test_path = os.path.join(TEST_PATH, DATA_FILE_NAME)

        # # data prepare
        data_train_fold1 = pd.read_csv(train_fold1_path)
        data_train_fold2 = pd.read_csv(train_fold2_path)
        data_train_fold3 = pd.read_csv(train_fold3_path)
        data_train_fold4 = pd.read_csv(train_fold4_path)
        data_train_fold5 = pd.read_csv(train_fold5_path)
        data = pd.concat([data_train_fold1, data_train_fold2, data_train_fold3, data_train_fold4, data_train_fold5])
        print(data.shape)
        data.to_csv(train_path, index=False)

        # data = pd.read_csv(train_path, nrows=50)
        data = pd.read_csv(train_path)
        print(data.shape)
        data['age-gender'] = data.apply(lambda x: str(x['age']) + '-' + str(x['gender']), axis=1)
        # print(data['age-gender'].value_counts())

        skf = StratifiedKFold(n_splits=5, random_state=1017, shuffle=True)

        for i, (train_index, valid_index) in enumerate(skf.split(data, data['age-gender'])):
            if (i+1) == k:
                print("FOLD | ", i + 1)
                print("###" * 35)
                data_train_kfold_path = os.path.join(MID_DATA_PATH, MID_DATA_TRAIN_FOLDK_FILE_NAME.format(k=str(i + 1)))
                data_valid_kfold_path = os.path.join(MID_DATA_PATH, MID_DATA_VALID_FOLDK_FILE_NAME.format(k=str(i + 1)))
                model_path = os.path.join(MODEL_PATH, MODEL_KFOLD_FILE_NAME.format(version=VERSION, k=str(i + 1)))
                print(model_path)

                ####### prepare data
                data_train_kfold = data.iloc[train_index]
                data_train_kfold = data_train_kfold.drop('age-gender', axis=1)
                data_train_kfold.to_csv(data_train_kfold_path, index=False)
                data_valid_kfold = data.iloc[valid_index]
                data_valid_kfold = data_valid_kfold.drop('age-gender', axis=1)
                data_valid_kfold.to_csv(data_valid_kfold_path, index=False)

                ####### prepare model
                model = build_model()
                if NUM_GPU>1:
                    model = multi_gpu_model(model, gpus=NUM_GPU) #每个gpu会跑BATCH_SIZE/2的数据

                # loss compile
                # optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=0.0)
                adam_optimizer = Adam(lr=1e-3, decay=5e-5, clipvalue=5)
                model.compile(optimizer=adam_optimizer,
                              loss='categorical_crossentropy', loss_weights=[0.9, 0.1],
                              metrics=['accuracy'])

                model.save(model_path)

                ####### train model
                earlystop = EarlyStopping(monitor='val_out_age_acc', min_delta=0, patience=2, verbose=1, mode='max')
                checkpoint = ModelCheckpoint(monitor='val_out_age_acc', mode='max', filepath=model_path, verbose=1,
                                             save_best_only=True)
                reducelearningrate = ReduceLROnPlateau(monitor='val_out_age_acc', factor=0.5, patience=1, verbose=1,
                                                       mode='max',
                                                       epsilon=0.0001, cooldown=0, min_lr=0)
                callback_list = [earlystop, checkpoint, reducelearningrate]

                hist = model.fit_generator(generate_from_file(data_train_kfold_path, BATCH_SIZE, 'train'),
                                           steps_per_epoch=calculate_steps(data_train_kfold.shape[0], BATCH_SIZE),
                                           epochs=TRAIN_EPOCH,
                                           validation_data=generate_from_file(data_valid_kfold_path, BATCH_SIZE, 'train'),
                                           validation_steps=calculate_steps(data_valid_kfold.shape[0], BATCH_SIZE),
                                           callbacks=callback_list, max_queue_size=2)

    def pred_model_kfold_k(k):
        """
            print_lines(train_path) #(719992, 11)
            print_lines(valid_path) #(180008, 11)
            print_lines(test_path) #(1000000, 9)
            :return:
            """
        from sklearn.model_selection import StratifiedKFold

        # path setting
        train_path = os.path.join(MID_DATA_PATH, MID_DATA_TRAIN_ALL_FILE_NAME)
        test_path = os.path.join(TEST_PATH, DATA_FILE_NAME)

        # data = pd.read_csv(train_path, nrows=50)
        data = pd.read_csv(train_path)
        print(data.shape)
        data['age-gender'] = data.apply(lambda x: str(x['age']) + '-' + str(x['gender']), axis=1)
        # print(data['age-gender'].value_counts())

        skf = StratifiedKFold(n_splits=5, random_state=1017, shuffle=True)
        sub_age = np.zeros((1000000, 10))
        sub_gender = np.zeros((1000000, 2))

        for i, (train_index, valid_index) in enumerate(skf.split(data, data['age-gender'])):
            if (i+1) == k:
                print("FOLD | ", i + 1)
                print("###" * 35)
                data_train_kfold_path = os.path.join(MID_DATA_PATH, MID_DATA_TRAIN_FOLDK_FILE_NAME.format(k=str(i + 1)))
                data_valid_kfold_path = os.path.join(MID_DATA_PATH, MID_DATA_VALID_FOLDK_FILE_NAME.format(k=str(i + 1)))
                model_path = os.path.join(MODEL_PATH, MODEL_KFOLD_FILE_NAME.format(version=VERSION, k=str(i + 1)))
                print(model_path)

                ####### prepare data
                # data_train_kfold = data.iloc[train_index]
                # data_train_kfold = data_train_kfold.drop('age-gender', axis=1)
                # data_train_kfold.to_csv(data_train_kfold_path, index=False)
                data_valid_kfold = data.iloc[valid_index]
                data_valid_kfold = data_valid_kfold.drop('age-gender', axis=1)
                data_valid_kfold.to_csv(data_valid_kfold_path, index=False)

                ####### prepare model
                model = build_model()
                if NUM_GPU > 1:
                    model = multi_gpu_model(model, gpus=NUM_GPU)  # 每个gpu会跑BATCH_SIZE/2的数据
                model.load_weights(model_path)


                #### pred test
                y_age, y_gender = model.predict_generator(generator=generate_from_file(test_path, 1000, 'test'),
                                                          steps=1000,
                                                          verbose=1, max_queue_size=2)
                sub_age += y_age
                sub_gender += y_gender

                #### pred valid
                pred_valid = model.predict_generator(
                    generator=generate_from_file(data_valid_kfold_path, BATCH_SIZE, 'train'),
                    steps=calculate_steps(data_valid_kfold.shape[0], BATCH_SIZE), verbose=1, max_queue_size=2)
                y_age_valid, y_gender_valid = pred_valid[0][:data_valid_kfold.shape[0]], pred_valid[1][:data_valid_kfold.shape[0]]


        # 处理test集的结果
        sub_age_avg = sub_age
        sub_gender_avg = sub_gender
        user_id = pd.read_csv(test_path, usecols=['user_id'], nrows=None)
        user_id['predicted_age'] = np.argmax(sub_age_avg, axis=1).astype(int) + 1
        user_id['predicted_gender'] = np.argmax(sub_gender_avg, axis=1).astype(int) + 1
        print(user_id['predicted_age'].value_counts())
        print(user_id['predicted_gender'].value_counts())
        user_id.to_csv(os.path.join(SUBMISSION_PATH, 'submission_%sfold%s.csv' % (VERSION,str(k))), index=False)
        for i in range(10):
            user_id['predicted_age_%d' % (i + 1)] = sub_age_avg[:, i]
        for i in range(2):
            user_id['predicted_gender_%d' % (i + 1)] = sub_gender_avg[:, i]
        user_id.to_csv(os.path.join(SUBMISSION_PATH, 'proba_%sfold%s.csv' % (VERSION, str(k))), index=False)

        # 处理valid集的结果
        user_id = pd.read_csv(data_valid_kfold_path, usecols=['user_id', 'age', 'gender'], nrows=None)
        user_id['predicted_age'] = np.argmax(y_age_valid, axis=1)[:len(user_id)].astype(int) + 1
        user_id['predicted_gender'] = np.argmax(y_gender_valid, axis=1)[:len(user_id)].astype(int) + 1
        user_id.to_csv(os.path.join(SUBMISSION_PATH, 'valid_%sfold%s.csv' % (VERSION, str(k))), index=False)

        # pred all
        for i in range(10):
            user_id['predicted_age_%d' % (i + 1)] = y_age_valid[:len(user_id), i]
        for i in range(2):
            user_id['predicted_gender_%d' % (i + 1)] = y_gender_valid[:len(user_id), i]
        user_id.to_csv(os.path.join(SUBMISSION_PATH, 'valid_proba_%sfold%s.csv' % (VERSION, str(k))), index=False)



    """根据mode选择执行的函数"""
    # check model
    model = build_model()
    print(model.summary())

    if MODE == "train_kfold_k":
        print(MODE, FOLDK)
        train_model_kfold_k(FOLDK)

    elif MODE == "pred_kfold_k":
        print(MODE, FOLDK)
        pred_model_kfold_k(FOLDK)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

