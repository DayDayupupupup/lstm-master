# -*- coding: utf-8 -*-
# run in py3 !!
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import time
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Convolution1D, Convolution2D, LSTM
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback

from keras.models import Model

from keras import initializers, layers
from keras.optimizers import SGD, Adadelta, Adam
from keras.regularizers import l1, l2
from keras import regularizers
import sys
import csv
sys.path.append('.')
from hist_figure import his_figures

if len(sys.argv) > 1:
    prefix = sys.argv[1]
else:
    prefix = time.time()


DATAPATH = './data/'
RESULT_PATH = './results/'
feature_num = 25
batch_num = 2
# batch_size = 32
batch_size = 512
SEQ_LENGTH = 20
STATEFUL = False

# scaler = preprocessing.MaxAbsScaler()
# scaler = preprocessing.StandardScaler()


def get_data(path_to_dataset='df.csv', sequence_length=SEQ_LENGTH, stateful=False, issplit = True):
    fold_index = 1
    ###
    dtypes = {'accelerator_x': 'float', 'accelerator_y': 'float', 'accelerator_z': 'float',
            'magnetic_x': 'float', 'magnetic_y': 'float', 'magnetic_z': 'float',
            'orientation_x': 'float', 'orientation_y': 'float', 'orientation_z': 'float',
            'gyroscope_x': 'float', 'gyroscope_y': 'float', 'gyroscope_z': 'float',
            'step': 'int'}
    # parse_dates = ['date']
    print(path_to_dataset)
    df = pd.read_csv(DATAPATH+path_to_dataset, header = 0,  dtype=dtypes ,quotechar='"',encoding="utf-8")
    # df = pd.read_csv(DATAPATH + path_to_dataset, header=0, dtype=dtypes,  encoding="utf-8")

    print(df.columns)
    print("*"*30)
    print(df['accelerator_x'])
    print("*"*30)
    # df = df[df['error'] >= 0]
    # df_test = pd.read_csv(DATAPATH+"test"+str(fold_index)+".csv", header = 0,  dtype=dtypes, parse_dates=parse_dates,encoding="utf-8")
    # def helper(x):
    #     split  = list(map(int, x.strip('[').strip(']').split(',')))
    #     d = {}
    #     for counter, value  in enumerate(split):
    #         k = str(len(split))+"-"+str(counter)
    #         d[k] = value
    #     return d
    # # df_train_temp = df_train['week'].apply(helper).apply(pd.Series)
    # df_week = df['week'].apply(helper).apply(pd.Series).as_matrix() #7
    # df_month = df['month'].apply(helper).apply(pd.Series).as_matrix() #12
    # df_year = df['year'].apply(helper).apply(pd.Series).as_matrix() #3

    # df_empty = df[[ 'super', 'com_date', 'error', 'numbers']].copy()
    # # print(df_empty)
    # df_super = df_empty.ix[:,[0]]
    # df_com_date = df_empty.ix[:,[1]]
    # df_error = df_empty.ix[:,[2]]
    # df_numbers = df_empty.ix[:,[3]]

    # ss_x = scaler
    # # ss_x = preprocessing.StandardScaler()
    # array_new = ss_x.fit_transform(df_empty.ix[:,[0]])
    # df_super = pd.DataFrame(array_new)

    # array_new = ss_x.fit_transform(df_empty.ix[:,[1]])
    # df_com_date = pd.DataFrame(array_new)

    # array_new = ss_x.fit_transform(df_empty.ix[:,[3]])
    # df_numbers = pd.DataFrame(array_new)

    # array_new = ss_x.fit_transform(df_empty.ix[:,[2]])
    # df_error = pd.DataFrame(array_new)

    # df_week = ss_x.fit_transform(df_week)
    # df_week = pd.DataFrame(df_week)

    # df_month = ss_x.fit_transform(df_month)
    # df_month = pd.DataFrame(df_month)


    # X_train = np.column_stack((df_super,df_com_date, df_numbers, df_week, df_month))
    X_train = np.column_stack((df['accelerator_x'],df['accelerator_y'],df['accelerator_z'],
                df['magnetic_x'],df['magnetic_y'],df['magnetic_z'],
                df['orientation_x'],df['orientation_y'],df['orientation_z'],
                df['gyroscope_x'],df['gyroscope_y'],df['gyroscope_z'],
                df['timestamp']
        ))
    Y_train = df['step'].as_matrix()
    print(Y_train.shape)
    y_arr = Y_train.T.tolist()
    # print(y_arr)

    try:
        draw_error_bar(y_arr[0])
    except Exception as e:
        print(e)
    if not issplit:
        return X_train, Y_train
    else:
        return split_CV(X_train, Y_train, sequence_length=SEQ_LENGTH, stateful=False)


def split_CV(X_train, Y_train, sequence_length=SEQ_LENGTH, stateful=False):
    """return ndarray
    """
    print(X_train.shape)
    print(Y_train.shape)
    result_x = []
    result_y = []
    for index in range(len(Y_train) - sequence_length):
        result_x.append(X_train[index: index + sequence_length])
        # result_y.append(Y_train[index: index + sequence_length])
        result_y.append(Y_train[index + sequence_length])
    X_train = np.array(result_x)
    Y_train = np.array(result_y)
    print(X_train.shape)#(705, 20, 24)
    print(Y_train.shape)#(705, 1)

    print('##################################################################')
    if stateful == True:
        # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1,shuffle=False)
        cp_X_train = X_train.copy()
        cp_Y_train = Y_train.copy()

        X_train = cp_X_train[:640,...]
        X_test = cp_X_train[640:,...]
        Y_train = cp_Y_train[:640,...]
        Y_test = cp_Y_train[640:,...]
        print(X_test.shape)#
        print(Y_test.shape)#
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1,shuffle=False)
        print('##################################################################')
    if stateful == False:
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1,shuffle=False)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1,shuffle=False)
    # print(X_train.shape)#(705, 20, 24)
    # print(Y_train.shape)#(705, 1)
    # train_x_disorder = X_train.reshape((X_train.shape[0],X_train.shape[1] , feature_num))
    # test_x_disorder = X_test.reshape((X_test.shape[0],X_test.shape[1], feature_num ))
    # X_val = X_val.reshape((X_val.shape[0], X_val.shape[1] , feature_num))
    # print(train_x_disorder.dtype)

    train_y_disorder = Y_train.reshape(-1, 1)
    test_y_disorder = Y_test.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    # print(test_y_disorder)
    return [X_train, train_y_disorder, X_test, test_y_disorder, X_val, Y_val] #ndarray


def LSTM2(X_train):
    model = Sequential()
    # layers = [1, 50, 100, 1]
    layers = [1, 20, 30, 1]
    if STATEFUL == False:
        model.add(LSTM(
                layers[1],
                input_shape=(X_train.shape[1], X_train.shape[2]),
                stateful=STATEFUL,
                return_sequences=True,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.01)))
    else:
        model.add(LSTM(
                layers[1],
                # input_shape=(X_train.shape[1], X_train.shape[2]),
                batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]),
                stateful=STATEFUL,
                return_sequences=True,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.01)))
    # model.add(Dropout(0.2))

    model.add(LSTM(
            layers[2],
            stateful=STATEFUL,
            return_sequences=False,
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.01)))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    model.add(Dense(
            layers[3]
            , kernel_initializer='he_normal'
            #, kernel_regularizer=l2(0.01),
            # activity_regularizer=l1(0.01)
            ))
    model.add(BatchNormalization())
    model.add(Activation("linear"))

    start = time.time()
    sgd = SGD(lr=1e-3, decay=1e-8, momentum=0.9, nesterov=True)
    ada = Adadelta(lr=1e-3, rho=0.95, epsilon=1e-6)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6, decay=1e-8)
    adam = Adam(lr=1e-3)
    # model.compile(loss="mse", optimizer=sgd)

    # try:
    #     model.load_weights("./lstm.h5")
    # except Exception as ke:
    #     print(str(ke))
    model.compile(loss="mse", optimizer=ada)
    print("Compilation Time : ", time.time() - start)
    return model

def draw_error_bar(y_array):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    x = list(range(len(y_array)))
    plt.bar(x, y_array, label='error',fc = 'r')
    #plt.legend(handles=[line1, line2,line3])
    plt.legend()
    plt.title('error bar')
    # plt.show()

    axes.grid()
    fig.tight_layout()
    fig.savefig(RESULT_PATH+str(batch_size)+str(SEQ_LENGTH)+'bar_error.png', dpi=300)


def draw_scatter(predicted, y_test, X_test, x_train, y_train, data_file):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    x = list(range(len(predicted)))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, y_test.T[0], width=width, label='truth',fc = 'y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, predicted, width=width, label='predict',fc = 'r')

    #plt.legend(handles=[line1, line2,line3])
    plt.legend()
    plt.title('lstm')
    # plt.show()

    axes.grid()
    fig.tight_layout()
    fig.savefig(RESULT_PATH+str(batch_size)+str(SEQ_LENGTH)+data_file+str(prefix)+'bar_lstm.png', dpi=300)

    fig = plt.figure()
    plt.scatter(y_test.T[0], predicted)
    # plt.plot(y_test.T[0], predicted, linewidth =0.3, color='red')
    plt.xlabel('truth')
    plt.ylabel('predict')
    # plt.show()
    fig.savefig(RESULT_PATH+str(batch_size)+str(SEQ_LENGTH)+data_file+str(prefix)+'_scatter_lstm.png', dpi=300)

def stat_metrics(X_test, y_test, predicted):
    predicted = np.reshape(predicted, y_test.shape[0])
    train_error =  np.abs(y_test - predicted)
    mean_error = np.mean(train_error)
    min_error = np.min(train_error)
    max_error = np.max(train_error)
    std_error = np.std(train_error)
    print(predicted)
    print(y_test.T[0])
    print(np.mean(X_test))

    print("#"*20)
    print(mean_error)
    print(std_error)
    print(max_error)
    print(min_error)
    print("#"*20)
    print(X_test[:,1])
    # 0.165861394194
    # ####################
    # 0.238853857898
    # 0.177678269353
    # 0.915951014937
        # 5.2530646691e-0
    pass

def run_network(model=None, data=None, data_file = 'df_dh.csv', isload_model = False, testonly = False):
    epochs = 300
    path_to_dataset = data_file
    sequence_length = SEQ_LENGTH

    if data is None:

        X_train, y_train, X_test, y_test, X_val, Y_val = get_data(sequence_length=sequence_length, stateful=STATEFUL, path_to_dataset=data_file)
    else:
        X_train, y_train, X_test, y_test, X_val, Y_val = data
    print('##################################################################')
    print(X_train[...,1])
    print(X_test.shape)
    if STATEFUL:
        X_test = X_test[:int(X_test.shape[0]/batch_size)*batch_size]
        y_test = y_test[:int(y_test.shape[0]/batch_size)*batch_size]
    print(X_test.shape)
    print(y_test.shape)
    print(X_test[:,:,1])

    if model is None:
        model = LSTM2(X_train)
        # print(model.get_config())
        if isload_model == True:
            try:
                model.load_weights("./lstm.h5")
            except Exception as ke:
                print(str(ke))
    if testonly == True:
        predicted = model.predict(X_test, verbose=1,batch_size=batch_size)
        predicted_arr = predicted.T.tolist()
        stat_metrics(X_test, y_test, predicted)
        draw_scatter(predicted_arr[0], y_test, X_test, X_train, y_train, data_file)

        return
    try:
        print("######################fit######################")
        early_stop=EarlyStopping(monitor='val_loss',patience=20)
        hist = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            # batch_size=512,
            nb_epoch=epochs, validation_data=(X_val, Y_val), callbacks=[],
            shuffle=False
             # shuffle=(not STATEFUL)
             )#, validation_split=0.05)
        print(model.get_config())
        if isload_model : model.save_weights("./lstm.h5")

        predicted = model.predict(X_test, verbose=1,batch_size=batch_size)
        #prediction_trans = scaler.inverse_transform(prediction)  xxxxxxx
        # prediction_trans = scaler.inverse_transform(predicted)
        # X_test_trans = scaler.inverse_transform(X_test)
        # y_test_trans = scaler.inverse_transform(y_test)
        # X_train_trans = scaler.inverse_transform(X_train)
        # y_train_trans = scaler.inverse_transform(y_train)
        stat_metrics(X_test, y_test, predicted)
    except KeyboardInterrupt as ke:
        print(str(ke))
        return model, y_test, 0
    try:
        predicted_df = pd.DataFrame(predicted)
        y_test_df = pd.DataFrame(y_test)
        # X_test_df = pd.DataFrame(X_test) #columns
        predicted_df.to_csv(DATAPATH+str(prefix)+data_file+str(batch_size)+str(SEQ_LENGTH)+"predicted_df.csv")
        y_test_df.to_csv(DATAPATH+str(prefix)+data_file+str(batch_size)+str(SEQ_LENGTH)+"y_test_df.csv")
        # X_test_df.to_csv(DATAPATH+data_file+"X_test_df.csv")
    except Exception as e:
        print("failed save predicted_df")
        raise e
    try:
        print("##############################################")
        print(predicted.shape)
        predicted_arr = predicted.T.tolist()
        # print(predicted_arr)
        draw_scatter(predicted_arr[0], y_test, X_test, X_train, y_train, data_file)
        his_figures(hist)
    except Exception as e:
        print("failed draw picture")
        raise e
    # print('Training duration (s) : ', time.time() - global_start_time)
    return model, y_test, predicted

def run_regressor(model=LSTM2, data=None, data_file = 'df_dh.csv', isload_model = True, testonly = False):
    epochs = 300
    path_to_dataset = data_file
    sequence_length = SEQ_LENGTH

    if data is None:

        X_train, y_train, X_test, y_test, X_val, Y_val = get_data(sequence_length=sequence_length, stateful=STATEFUL, path_to_dataset=data_file)
    else:
        X_train, y_train, X_test, y_test, X_val, Y_val = data

    if STATEFUL:
        X_test = X_test[:int(X_test.shape[0]/batch_size)*batch_size]
        y_test = y_test[:int(y_test.shape[0]/batch_size)*batch_size]

    estimator = KerasRegressor(build_fn=lambda x=X_train:model(x))

    # if testonly == True:
    #     # predicted = model.predict(X_test, verbose=1,batch_size=batch_size)
    #     prediction = estimator.predict(X_test)

    #     stat_metrics(X_test, y_test, prediction)
    #     draw_scatter(predicted_arr[0], y_test, X_test, X_train, y_train, data_file)
    #     return

    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=20)
    checkpoint = ModelCheckpoint("./lstm.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True)
    ################
    hist = estimator.fit(X_train, y_train, validation_data=(X_val, Y_val), callbacks=[ checkpoint], epochs=epochs, batch_size=batch_size, verbose=1)

    # prediction = estimator.predict(X_test)
    score = mean_squared_error(y_test, estimator.predict(X_test))
    estimator_score = estimator.score(X_test, y_test)
    print(score)

    prediction = estimator.predict(X_test)
    # invert predictions
    # prediction_trans = scaler.inverse_transform(prediction)
    # X_test_trans = scaler.inverse_transform(X_test)
    # y_test_trans = scaler.inverse_transform(y_test)
    # X_train_trans = scaler.inverse_transform(X_train)
    # y_train_trans = scaler.inverse_transform(y_train)


    print(prediction)
    print(X_test)
    print("##############################################")
    # predicted_arr = prediction.T.tolist()
    # print(predicted_arr)
    draw_scatter(prediction, y_test, X_test, X_train, y_train, data_file)
    his_figures(hist)

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_val, Y_val = get_data(sequence_length=SEQ_LENGTH, stateful=STATEFUL, path_to_dataset='step_7_4__7_10.csv')
    run_regressor(data = [X_train, y_train, X_test, y_test, X_val, Y_val],data_file = 'step_7_4__7_10.csv', isload_model=False)
    # bX_train, by_train, bX_test, by_test, bX_val, bY_val = get_data(sequence_length=SEQ_LENGTH, stateful=STATEFUL, path_to_dataset='bombNew.csv')
    #print(bX_test.shape)
    print(X_test.shape)
    # run_regressor(data = [X_train, y_train, bX_test, by_test, X_val, Y_val], isload_model=True, testonly = True)
    # run_network(data_file = 'bomb.csv', isload_model=True)

# stock_predict tf
# https://github.com/LouisScorpio/datamining/blob/master/tensorflow-program/rnn/stock_predict/stock_predict_2.py

# boston tf
# https://blog.csdn.net/baixiaozhe/article/details/54410313

########### consume predict keras
# http://www.cnblogs.com/arkenstone/p/5794063.html

# bike number predict keras
# http://resuly.me/2017/08/16/keras-rnn-tutorial/#%E4%BB%BB%E5%8A%A1%E6%8F%8F%E8%BF%B0

# Multivariate Time Series Forecasting with LSTMs in Keras
# https://zhuanlan.zhihu.com/p/28746221
