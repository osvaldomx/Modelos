# coding: UTF-8

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from keras.layers import Dropout, Dense, InputLayer, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.regularizers import l1_l2

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


    model = Sequential()
    model.add(InputLayer(input_shape=(10, 206)))

    model.add(Conv1D(16, 10, activation='elu', activity_regularizer=l1_l2(.01, .01)))
    model.add(BatchNormalization())
    model.add(Dropout(.25))
    model.add(Reshape((1, 4, 4)))

    model.add(Conv2D(4, (8, 8), padding='same', activation='elu', data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Dropout(.25))

    model.add(Conv2D(4, (8, 4), padding='same', activation='elu', data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Dropout(.25))

    model.add(Flatten())

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #class_weight = {0:.2, 1:.8}
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)

    #model.fit(X_train, y_train_cat,
    #          validation_data=(X_val, y_val_cat),
    #          epochs=50,
    #          class_weight=class_weight)

    model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), epochs=100)

    print(model.evaluate(X_test, y_test_cat))

    predict = model.predict_classes(X_test)

    fpr, tpr, threshold = roc_curve(y_test, predict)
    area = auc(fpr, tpr)

    print('\n')
    #print'balancing:{0}'.format(class_weight)
    print'y_test = {0}'.format(y_test.sum())
    print'predict = {0}'.format(predict.sum())
    print'auc = {0}'.format(area)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.4f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: EEGNet with P300 data')
    plt.legend(loc="lower right")
    plt.show()
