# coding: UTF-8
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

from functools import reduce

from keras.layers import Dropout, Dense, InputLayer, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn.preprocessing import scale
from sklearn.metrics import auc, roc_curve
from sklearn.utils import resample

from scipy.stats import zscore

np.random.seed(123)


def reformatInput(data, labels, indices):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.
    """

    trainIndices = indices[0][len(indices[1]):]
    validIndices = indices[0][:len(indices[1])]
    testIndices = indices[1]
    # Shuffling training data
    # shuffledIndices = np.random.permutation(len(trainIndices))
    # trainIndices = trainIndices[shuffledIndices]
    if data.ndim == 3:
        return [(data[trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]
    elif data.ndim == 5:
        return [(data[:, trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[:, validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[:, testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]


if __name__ == '__main__':

    data = sio.loadmat('p300_data/dataAcs')

    X = np.concatenate((data['dataCalor'], data['dataCarino'], data['dataSushi']))
    X = np.swapaxes(X, 2, 1)

    Y = np.concatenate((data['labelCalor'][:, 1:2],
                        data['labelCarino'][:, 1:2],
                        data['labelSushi'][:, 1:2]))
    np.place(Y, Y < 0, 0)
    Y = np.squeeze(Y)

    X, Y = resample(X, Y, random_state=1)

    X = zscore(X, axis=None)

    subj_nums = np.load('p300_data/subj.npy')

    # Leave-Subject-Out cross validation
    fold_pairs = []
    for i in np.unique(subj_nums):
        ts = subj_nums == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        np.random.shuffle(tr)  # Shuffle indices
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))

    fold = fold_pairs[2]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(X, Y, fold)

    rsvp = Sequential()

