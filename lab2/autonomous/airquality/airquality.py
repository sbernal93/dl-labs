"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:47 

"""

from __future__ import print_function
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Flatten 
from keras.layers import Dropout 
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import os
import tensorflow as tf
import json
import argparse
from time import time
import numpy as np
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

__author__ = 'bejar'


def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)

def lagged_vector(data, lag=1, ahead=0):
    """
    Returns a vector with columns that are the steps of the lagged time series
    Last column is the value to predict

    Because arrays start at 0, Ahead starts at 0 but actually means one step ahead

    :param data:
    :param lag:
    :return:
    """
    lvect = []
    for i in range(lag):
        lvect.append(data[i: -lag - ahead + i])
    lvect.append(data[lag + ahead:])

    return np.stack(lvect, axis=1)


def lagged_matrix(data, lag=1, ahead=0):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []

    for i in range(lag):
        lvect.append(data[i: -lag - ahead + i, :])
    lvect.append(data[lag + ahead:, :])
    return np.stack(lvect, axis=1)


def _generate_dataset_one_var(data, datasize, testsize, lag=1, ahead=1):
    """
    Generates dataset assuming only one variable for prediction
    Here ahead starts at 1 (I know it is confusing)

    :return:
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # print('DATA Dim =', data.shape)

    wind_train = data[:datasize, :]
    # print('Train Dim =', wind_train.shape)

    train = lagged_vector(wind_train, lag=lag, ahead=ahead - 1)

    train_x, train_y = train[:, :lag], train[:, -1:, 0]

    wind_test = data[datasize:datasize + testsize, 0].reshape(-1, 1)
    test = lagged_vector(wind_test, lag=lag, ahead=ahead - 1)

    test_x, test_y = test[:, :lag], test[:, -1:, 0]

    return train_x, train_y, test_x, test_y


def _generate_dataset_multi_var(data, datasize, testsize, datavars, lag=1, ahead=1):

    #scales the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # print('DATA Dim =', data.shape)

    train_data = data[:datasize, :]
    # print('Train Dim =', wind_train.shape)

    # the matrix is generated with 3 dimensions: (datasize, lag +1 (series), datavars)
    train = lagged_matrix(train_data, lag=lag, ahead=ahead - 1)

    # trainx is without one series 
    # trainy is one series with the attributes needed (datavars)
    train_x, train_y = train[:, :lag], train[:, -1]

    # as well as last time, the reshape here isnt needed
    test_data = data[datasize:datasize + testsize, :]
    test = lagged_matrix(test_data, lag=lag, ahead=ahead - 1)

    test_x, test_y = test[:, :lag], test[:, -1]

    return train_x, train_y, test_x, test_y

def generate_dataset(config, ahead=1, data_path=None):
    """
    Generates the dataset for training, test and validation

    :param ahead: number of steps ahead for prediction

    :return:
    """
    dataset = config['dataset']
    datanames = config['datanames']
    datasize = config['datasize']
    testsize = config['testsize']
    vars = config['vars']
    lag = config['lag']

    airq = {}

    # Reads numpy arrays for all sites and keep only selected columns

    aqdata = np.load(data_path + 'LondonAQ.npz')
    for d in datanames:
        airq[d] = aqdata[d]
        if vars is not None:
            airq[d] = airq[d][:, vars]

    if dataset == 0:
        return _generate_dataset_one_var(airq[datanames[0]][:, 0].reshape(-1, 1), datasize, testsize,
                                         lag=lag, ahead=ahead)
    # Just add more options to generate datasets with more than one variable for predicting one value
    # or a sequence of values

    # airq[datanames[0]][:, 0:3] for getting from one var to another
    # reshape necessary? 4 prev case yields same matrix if (-1,3)
    # datanames[0]=GEltham

    # airq has all the data with all the datanames ("GEltham", "GWesthorne", "BSladeGreen",  "GWoolwich")
    # airq[datanames[0]] is only GEltham, airq[datanames[0]] also has 5 columns, one for each
    # of the variables, in the one var, it only uses PM10. For multi var, we use all defined 
    # in dataset, so we grab all the columns from 0 to the dataset value hence:
    # airq[datanames[0]][:, 0:dataset]. the .reshape in one var is there because the data 
    # is of shape: (43848,) which is a vector, after reshape its (43848, 1) which puts it 
    # all in a column making it a matrix. For multiple variables the reshape isnt needed
    # as a reshape(-1, 3) returns the same matrix. 

    if dataset > 0:
        return _generate_dataset_multi_var(airq[datanames[0]][:, 0:dataset+1], datasize, testsize,
                                      dataset, lag=lag, ahead=ahead)

    raise NameError('ERROR: No such dataset type')


def architecture(neurons, drop, nlayers, activation, activation_r, rnntype, num_classes=0, impl=1):
    """
    RNN architecture

    :return:
    """
    #if num_classes == 0:
    #    num_classes=1

    RNN = LSTM if rnntype == 'LSTM' else GRU
    model = Sequential()
    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r))
    else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                      recurrent_activation=activation_r, implementation=impl))

    model.add(Dense(num_classes+1))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=True)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 1  # implementation 0 is deprecated for newer tensorflow versions

    config = load_config_file(args.config)
    ############################################
    # Data

    ahead = config['data']['ahead']

    if args.verbose:
        print('-----------------------------------------------------------------------------')
        print('Steps Ahead = %d ' % ahead)

    # Modify conveniently with the path for your data
    aq_data_path = './'

    train_x, train_y, test_x, test_y = generate_dataset(config['data'],
     ahead=ahead, data_path=aq_data_path)

    ############################################
    # Model

    model = architecture(neurons=config['arch']['neurons'],
                         drop=config['arch']['drop'],
                         nlayers=config['arch']['nlayers'],
                         activation=config['arch']['activation'],
                         activation_r=config['arch']['activation_r'],
                         rnntype=config['arch']['rnn'],
                         num_classes=config['data']['dataset'],
                         impl=impl)
    if args.verbose:
        model.summary()
        print('lag: ', config['data']['lag'],
              '/Neurons: ', config['arch']['neurons'],
              '/Layers: ', config['arch']['nlayers'],
              '/Activations:', config['arch']['activation'], config['arch']['activation_r'])
        print('Tr:', train_x.shape, train_y.shape, 'Ts:', test_x.shape, test_y.shape)
        print()

    ############################################
    # Training

    optimizer = config['training']['optimizer']

    if optimizer == 'rmsprop':
        if 'lrate' in config['training']:
            optimizer = RMSprop(lr=config['training']['lrate'])
        else:
            optimizer = RMSprop(lr=0.001)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    cbacks = []

    if args.tboard:
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        cbacks.append(tensorboard)

    if args.best:
        modfile = './model%d.h5' % int(time())
        mcheck = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
        cbacks.append(mcheck)


    model.fit(train_x, train_y, batch_size=config['training']['batch'],
              epochs=config['training']['epochs'],
              validation_data=(test_x, test_y),
              verbose=verbose, callbacks=cbacks)

    ############################################
    # Results

    if args.best:
        model = load_model(modfile)

    score = model.evaluate(test_x, test_y, batch_size=config['training']['batch'], verbose=0)

    print()
    print('MSE test= ', score)
    print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))
    test_yp = model.predict(test_x, batch_size=config['training']['batch'], verbose=0)
    r2test = r2_score(test_y, test_yp)
    r2pers = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])
    print('R2 test= ', r2test)
    print('R2 test persistence =', r2pers)

    resfile = open('result-%s.txt' % config['data']['datanames'][0], 'a')
    resfile.write('DATAS= %d, LAG= %d, AHEAD= %d, RNN= %s, NLAY= %d, NNEUR= %d, DROP= %3.2f, ACT= %s, RACT= %s, '
                  'OPT= %s, R2Test = %3.5f, R2pers = %3.5f\n' %
                  (config['data']['dataset'],
                   config['data']['lag'],
                   config['data']['ahead'],
                   config['arch']['rnn'],
                   config['arch']['nlayers'],
                   config['arch']['neurons'],
                   config['arch']['drop'],
                   config['arch']['activation'],
                   config['arch']['activation_r'],
                   config['training']['optimizer'],
                   r2test, r2pers
                   ))
    resfile.close()

    # Deletes the model file
    if args.best:
        try:
            os.remove(modfile)
        except OSError:
            pass
