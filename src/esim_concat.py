import os
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.activations import softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import sys 

used_cols = ['creative_id', 'advertiser_id', 'ad_id', 'product_id', 'industry']
full_cols_dict = {'user_id': 0, 'age': 1, 'gender': 2, 'time': 3, 'creative_id': 4, 'click_times': 5, 'ad_id': 6, 'product_id': 7, 'product_category': 8, 'advertiser_id': 9, 'industry': 10}
full_cols_dict_test = {'user_id': 0, 'time': 1, 'creative_id': 2, 'click_times': 3, 'ad_id': 4, 'product_id': 5, 'product_category': 6, 'advertiser_id': 7, 'industry': 8}

W2V_PATH = '../models/w2v/'
W2V_FILE_NAME = 'w2v_embeddings_{col}_p20200628.npy'
FEAT_PATH = '../data/semi/{}/{}'
MODEL_PATH = '../models/esim_concat/'
PRED_PATH = '../result/'
MAX_SEQ_LEN = 175
LATENT_DIM = 200
BATCH_SIZE = 1500
TRAIN_EPOCH = 20
DENSE_INPUT_DIM = 288
HIDDEN_LAYERS = [256, 128, 64]

W2V_VECTOR = dict()
for col in used_cols:
    W2V_VECTOR[col] = np.load(os.path.join(W2V_PATH, W2V_FILE_NAME.format(col=col)))
    print(col, W2V_VECTOR[col].shape)

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
        if len(minibatch[i]) > maxlen:
            res.append(minibatch[i][:maxlen, :])
        else:
            res.append(np.concatenate([minibatch[i], np.zeros([maxlen-lens[i]]+dim, dtype=np.int32)], axis=0))
    return np.asarray(res)

def process_seq_line(line, stage='train'):
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

def generate_from_file(seq_file_path_list=[], dense_file_path_list=[], batch_size=64, stage='train'):
    X, Y = [[] for _ in range(len(used_cols) + 1)], [[], []]
    while True:
        for seq_path, dense_path in zip(seq_file_path_list, dense_file_path_list):
            with open(seq_path, 'r') as fin_s, open(dense_path, 'r') as fin_d:
                fin_s.readline(), fin_d.readline()  # header
                while True:
                    line_s = fin_s.readline().rstrip()
                    line_d = fin_d.readline().rstrip()
                    if not line_s or not line_d:
                        break
                    xs, ys = process_seq_line(line_s, stage)
                    if xs is None or ys is None:
                        continue
                    for i in range(len(xs)):
                        X[i].append(xs[i])
                    for i in range(len(ys)):
                        Y[i].append(ys[i])

                    xd = np.array(line_d.split(','), dtype=np.float32)
                    X[-1].append(xd)

                    if len(X[0]) >= batch_size:
                        yield [padding(x, MAX_SEQ_LEN) for x in X[:-1]] + [np.array(X[-1])], [one_hot_encode(Y[0], 10), one_hot_encode(Y[1], 2)]
                        X, Y = [[] for _ in range(len(used_cols) + 1)], [[], []]





def build_model():
    input_dense = Input((DENSE_INPUT_DIM, ), name='input_dense')
    emb = input_dense
    for i in range(len(HIDDEN_LAYERS)):
        emb = Dense(HIDDEN_LAYERS[i], activation='relu', name='dnn_%d' % (i+1))(emb)
        if i < len(HIDDEN_LAYERS) - 1:
            emb = Dropout(0.5, name='dnn_drop_%d' % (i+1))(emb)
    output_dense = emb

    # each [Batch_size, T, 200]
    input_tensor1 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input1')
    input_tensor2 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input2')
    input_tensor3 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input3')
    input_tensor4 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input4')
    input_tensor5 = Input((MAX_SEQ_LEN, LATENT_DIM), name='input5')

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

    final_embedding = concatenate \
        ([final_embedding_sequence_q1, final_embedding_sequence_q2, final_embedding_sequence_q3,
          final_embedding_sequence_q4, final_embedding_sequence_q5])

    rnn_layer_q1 = LSTM(LATENT_DIM, return_sequences=True)(final_embedding)
    rnn_layer_q2 = LSTM(LATENT_DIM, return_sequences=True)(final_embedding)

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

    v_q1_i = LSTM(LATENT_DIM, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(m_q1)
    v_q2_i = LSTM(LATENT_DIM, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(m_q2)

    avgpool_q1 = GlobalAveragePooling1D()(v_q1_i)
    avgpool_q2 = GlobalAveragePooling1D()(v_q2_i)
    maxpool_q1 = GlobalMaxPooling1D()(v_q1_i)
    maxpool_q2 = GlobalMaxPooling1D()(v_q2_i)

    merged_q1 = concatenate([avgpool_q1, maxpool_q1])
    merged_q2 = concatenate([avgpool_q2, maxpool_q2])

    final_v = BatchNormalization()(concatenate([merged_q1, merged_q2]))

    output_age = Dense(units=64, activation='relu')(final_v)
    output_age = BatchNormalization()(output_age)
    output_age = Dropout(0.2)(output_age)
    output_age = concatenate([output_age, output_dense])
    y_age = Dense(10, activation='softmax', name='out_age')(output_age)

    output_gender = Dense(units=64, activation='relu')(final_v)
    output_gender = BatchNormalization()(output_gender)
    output_gender = Dropout(0.2)(output_gender)
    output_gender = concatenate([output_gender, output_dense])
    y_gender = Dense(2, activation='softmax', name='out_gender')(output_gender)

    model = Model(inputs=[input_tensor1, input_tensor2, input_tensor3, input_tensor4, input_tensor5, input_dense], outputs=[y_age, y_gender])

    return model


def train_model(train_seq_paths, train_dense_paths,
                valid_seq_paths, valid_dense_paths,
                init_path=None, save_path=None):

    TRAIN_NUM = sum([len(pd.read_csv(p, usecols=['gender'])) for p in train_seq_paths])
    VALID_NUM = sum([len(pd.read_csv(p, usecols=['gender'])) for p in valid_seq_paths])
    print('train_size=%d, valid_size=%d' % (TRAIN_NUM, VALID_NUM))

    model = build_model()
    model.summary()

    if init_path is not None:
        model.load_weights(init_path)

    adam_optimizer = Adam(lr=1e-3, decay=5e-5, clipvalue=5)
    model.compile(optimizer=adam_optimizer,
                  loss='categorical_crossentropy', loss_weights=[0.7, 0.3],
                  metrics=['accuracy'])

    earlystop = EarlyStopping(monitor='val_out_age_acc', min_delta=0, patience=2, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(monitor='val_out_age_acc', mode='max', filepath=save_path, verbose=1,
                                 save_best_only=True)
    reducelearningrate = ReduceLROnPlateau(monitor='val_out_age_acc', factor=0.5, patience=1, verbose=1, mode='max',
                                           epsilon=0.0001, cooldown=0, min_lr=0)
    callback_list = [earlystop, checkpoint, reducelearningrate]

    model.fit_generator(generate_from_file(train_seq_paths, train_dense_paths, BATCH_SIZE, 'train'),
                        steps_per_epoch=TRAIN_NUM // BATCH_SIZE,
                        # steps_per_epoch=40,
                        validation_data=generate_from_file(valid_seq_paths, valid_dense_paths, BATCH_SIZE, 'train'),
                        validation_steps=VALID_NUM // BATCH_SIZE,
                        # validation_steps=40,
                        callbacks=callback_list, max_queue_size=2,
                        epochs=TRAIN_EPOCH,)

    if save_path is not None:
        model.save_weights(save_path)


def predict_valid(valid_fold_id, valid_seq_paths, valid_dense_paths, model_path):

    model = build_model()
    model.load_weights(model_path)
    print("load model from %s" % model_path)

    user_id = pd.concat([pd.read_csv(f, usecols=['user_id', 'age', 'gender'], nrows=None) for f in valid_seq_paths], 0)
    y_age_prob, y_gender_prob = model.predict_generator(generator=generate_from_file(valid_seq_paths, valid_dense_paths, 1500, 'train'),
                                                        steps=len(user_id) // 1500 + 1, verbose=1, max_queue_size=2)

    for i in range(10):
        user_id['predicted_age_%d' % (i+1)] = y_age_prob[:len(user_id), i]
    for i in range(2):
        user_id['predicted_gender_%d' % (i+1)] = y_gender_prob[:len(user_id), i]
    user_id['predicted_age'] = np.argmax(y_age_prob[:len(user_id)], 1) + 1
    user_id['predicted_gender'] = np.argmax(y_gender_prob[:len(user_id)], 1) + 1

    user_id.to_csv(os.path.join(PRED_PATH, 'valid_prob_%d_%s.csv' % (valid_fold_id, VERSION)))

    acc_age = accuracy_score(user_id.age.values, user_id.predicted_age.values)
    acc_gender = accuracy_score(user_id.gender.values, user_id.predicted_gender.values)
    print("valid fold%d acc = %.4f + %.4f = %.4f" % (valid_fold, acc_age, acc_gender, acc_age + acc_gender))


def predict_test(valid_fold_id, test_seq_paths, test_dense_paths, model_path):

    model = build_model()
    model.load_weights(model_path)
    print("load model from %s" % model_path)

    user_id = pd.concat([pd.read_csv(f, usecols=['user_id'], nrows=None) for f in test_seq_paths], 0)
    y_age_prob, y_gender_prob = model.predict_generator(generator=generate_from_file(test_seq_paths, test_dense_paths, 1500, 'test'),
                                                        steps=len(user_id) // 1500 + 1, verbose=1, max_queue_size=2)

    for i in range(10):
        user_id['predicted_age_%d' % (i+1)] = y_age_prob[:len(user_id), i]
    for i in range(2):
        user_id['predicted_gender_%d' % (i+1)] = y_gender_prob[:len(user_id), i]
    user_id['predicted_age'] = np.argmax(y_age_prob[:len(user_id)], 1) + 1
    user_id['predicted_gender'] = np.argmax(y_gender_prob[:len(user_id)], 1) + 1

    user_id.to_csv(os.path.join(PRED_PATH, 'test_prob_%d_%s.csv' % (valid_fold_id, VERSION)))                                                     


if __name__ == '__main__':
    # python esim_concat.py train_kfold_k 1 p20200713
    mode = str(sys.argv[1])
    VERSION = str(sys.argv[3])
    print("mode = %s" % mode)

    if mode == 'train_kfold_k':
        # only train one-fold
        valid_fold = int(sys.argv[2])
        print("valid_fold = %d" % valid_fold)

        train_seq_file_paths = [FEAT_PATH.format('folds/fold_%d' % fold, 'origin_feature_reindexed_fromzero.csv') for
                                fold in np.arange(5) + 1 if fold != valid_fold]
        train_dense_file_paths = [FEAT_PATH.format('folds/fold_%d' % fold, 'sklearn_pred_feat.csv') for fold in
                                  np.arange(5) + 1 if fold != valid_fold]

        valid_seq_file_paths = [FEAT_PATH.format('folds/fold_%d' % valid_fold, 'origin_feature_reindexed_fromzero.csv')]
        valid_dense_file_paths = [FEAT_PATH.format('folds/fold_%d' % valid_fold, 'sklearn_pred_feat.csv')]

        model_path = os.path.join(MODEL_PATH, "esim_concat_fold%d_%s.h" % (valid_fold, VERSION))
        train_model(train_seq_file_paths, train_dense_file_paths, valid_seq_file_paths, valid_dense_file_paths, None,
                    model_path)

    elif mode == 'train_kfolds':
        # train k-folds
        for valid_fold in np.arange(5) + 1:
            print("valid_fold = %d" % valid_fold)

            train_seq_file_paths = [FEAT_PATH.format('folds/fold_%d' % fold, 'origin_feature_reindexed_fromzero.csv')
                                    for fold in np.arange(5) + 1 if fold != valid_fold]
            train_dense_file_paths = [FEAT_PATH.format('folds/fold_%d' % fold, 'sklearn_pred_feat.csv') for fold in
                                      np.arange(5) + 1 if fold != valid_fold]

            valid_seq_file_paths = [
                FEAT_PATH.format('folds/fold_%d' % valid_fold, 'origin_feature_reindexed_fromzero.csv')]
            valid_dense_file_paths = [FEAT_PATH.format('folds/fold_%d' % valid_fold, 'sklearn_pred_feat.csv')]

            model_path = os.path.join(MODEL_PATH, "esim_concat_fold%d.h" % valid_fold)
            train_model(train_seq_file_paths, train_dense_file_paths, valid_seq_file_paths, valid_dense_file_paths,
                        None, model_path)

    elif mode == 'predict_kfold_k':
        # only predict one-fold
        valid_fold = int(sys.argv[2])
        print("valid_fold = %d" % valid_fold)

        valid_seq_file_paths = [FEAT_PATH.format('folds/fold_%d' % valid_fold, 'origin_feature_reindexed_fromzero.csv')]
        valid_dense_file_paths = [FEAT_PATH.format('folds/fold_%d' % valid_fold, 'sklearn_pred_feat.csv')]
        model_path = os.path.join(MODEL_PATH, "esim_concat_fold%d_%s.h" % (valid_fold, VERSION))
        
        predict_valid(valid_fold, valid_seq_file_paths, valid_dense_file_paths, model_path)

        test_seq_file_paths = [FEAT_PATH.format('test_preliminary', 'origin_feature_reindexed_fromzero.csv')]
        test_dense_file_paths = [FEAT_PATH.format('test_preliminary', 'sklearn_pred_feat.csv')]

        predict_test(valid_fold, test_seq_file_paths, test_dense_file_paths, model_path)

