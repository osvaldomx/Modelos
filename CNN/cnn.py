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

from utils import reformatInput, cart2sph, augment_EEG, pol2cart

from sklearn.preprocessing import scale
from sklearn.metrics import auc, roc_curve

np.random.seed(123)


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, np.pi / 2 - elev)


def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    # assert features.shape[1] % nElectrodes == 0
    n_colors = int(features.shape[1] / nElectrodes)
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in xrange(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        print'Interpolating {0}/{1}\r'.format(i + 1, nSamples, end='\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]


if __name__ == '__main__':
    # Load electrode locations
    print('Loading data...')
    locs_3d = sio.loadmat('p300_data/Neuroscan_locs_orig.mat')['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    data = sio.loadmat('p300_data/dataAcs.mat')

    feats = np.concatenate((data['dataCalor'], data['dataCarino'], data['dataSushi']))
    feats = np.swapaxes(feats, 2, 1)
    feats = feats.reshape((feats.shape[0], -1))

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

    labels = np.concatenate((data['labelCalor'][:, 1:2],
                        data['labelCarino'][:, 1:2],
                        data['labelSushi'][:, 1:2]))
    np.place(labels, labels < 0, 0)
    labels = np.squeeze(labels)

    fold = fold_pairs[2]
    batch_size = 32
    num_epochs = 5

    num_classes = len(np.unique(labels))

    # CNN Mode
    imsize = 32
    # Find the average response over time windows
    av_feats = reduce(lambda x, y: x + y, [feats[:, i * 206:(i + 1) * 206] for i in range(feats.shape[1] / 206)])
    av_feats = av_feats / (feats.shape[1] / 206)

    images = gen_images(np.array(locs_2d), av_feats, imsize, normalize=False)
    # images = np.load('images.npy')
    np.save('images', images)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)
    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')

    print('Building model...')

    network = Sequential()

    network.add(InputLayer(input_shape=(3, imsize, imsize)))

    network.add(Conv2D(32, 3, padding='same', data_format='channels_first', activation='relu'))
    network.add(Conv2D(32, 3, padding='same', data_format='channels_first', activation='relu'))
    network.add(Conv2D(32, 3, padding='same', data_format='channels_first', activation='relu'))
    network.add(Conv2D(32, 3, padding='same', data_format='channels_first', activation='relu'))

    network.add(MaxPooling2D(pool_size=(2, 2)))

    network.add(Conv2D(64, 3, padding='same', data_format='channels_first', activation='relu'))
    network.add(Conv2D(64, 3, padding='same', data_format='channels_first', activation='relu'))

    network.add(MaxPooling2D(pool_size=(2, 2)))

    network.add(Conv2D(128, 3, padding='same', data_format='channels_first', activation='relu'))

    network.add(MaxPooling2D(pool_size=(2, 2)))

    network.add(Flatten())

    network.add(Dropout(0.5))
    network.add(Dense(256, activation='relu'))

    network.add(Dropout(0.5))
    network.add(Dense(num_classes, activation='sigmoid'))

    network.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    # Class labels should start from 0
    print('Training the CNN Model...')

    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)

    #class_weights = {0: .2, 1: .8}

    network.fit(X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=num_epochs)
    #            class_weight=class_weights)

    network.save('cnn_p300.h5')
    network.save_weights('weights_cnn_p300.h5')

    print(network.evaluate(X_test, y_test_cat))

    predict = network.predict_classes(X_test)

    fpr, tpr, threshold = roc_curve(y_test, predict)
    area = auc(fpr, tpr)

    print('\n')
    #print'balancing:{0}'.format(class_weights)
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
    plt.title('Receiver operating characteristic: CovNet with P300 data')
    plt.legend(loc="lower right")
    plt.show()

    print('Done!')
