# coding: utf-8
import numpy as np
import scipy.io as sio

from scipy.interpolate import griddata

from functools import reduce

from keras.models import Model
from keras.layers import Dropout, Dense, Input, Flatten, Permute, Reshape, concatenate
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.optimizers import adam
from keras.utils import plot_model, to_categorical

from utils import cart2sph, pol2cart, augment_EEG, reformatInput

from sklearn.preprocessing import scale

from multiprocessing import Process


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


def build_cnn(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32,
              imsize=32, n_colors=3):
    """
    
    :param input_var: 
    :param w_init: 
    :param n_layers: 
    :param n_filters_first: 
    :param imsize: 
    :param n_colors: 
    :return: 
    """
    weights = []  # Keeps the weights for all layers
    count = 0

    # Input layer
    network = Input(shape=(n_colors, imsize, imsize), tensor=input_var)

    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv2D(n_filters_first * (2 ** i), (3, 3), padding='same',
                             data_format='channels_first')(network)
            count += 1
            #weights.append(network.weights)
        network = MaxPooling2D(pool_size=(2, 2))(network)

    return network


if __name__ == '__main__':
    # Load electrode locations
    print('Loading data...')
    locs = sio.loadmat('sample_data/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    feats = sio.loadmat('sample_data/FeatureMat_timeWin.mat')['features']
    subj_nums = np.squeeze(sio.loadmat('sample_data/trials_subNums.mat')['subjectNum'])
    # Leave-Subject-Out cross validation
    fold_pairs = []
    for i in np.unique(subj_nums):
        ts = subj_nums == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        np.random.shuffle(tr)  # Shuffle indices
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))

    # Conv-LSTM Mode
    print('Generating images for all time windows...')
    #images_timewin = np.array(
    #    [gen_images(np.array(locs_2d),
    #                feats[:, i * 192:(i + 1) * 192], 32,
    #                normalize=False)
    #     for i in range(feats.shape[1] / 192)])
    images_timewin = np.load('sample_data/images_timewin.npy')

    num_classes = 4
    grad_clip = 100
    imsize = 32
    n_colors = 3
    n_timewin = 3
    labels = np.squeeze(feats[:, -1]) - 1
    images = images_timewin
    fold = fold_pairs[2]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)
    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')

    print('Building LSTM-Conv Model')
    main_input = Input(shape=(None, 3, imsize, imsize))

    convnets = []
    w_init = None
    # Build 7 parallel CNNs with shared weights
    #for i in range(n_timewin):
    #    main_input = build_cnn(main_input[i], imsize=imsize, n_colors=n_colors)
    #    convnets.append(Flatten()(main_input))

    in1 = Input(shape=(3,imsize,imsize))
    net1 = Conv2D(32, 3, padding='same', data_format='channels_first')(in1)
    net1 = Conv2D(32, 3, padding='same', data_format='channels_first')(net1)
    net1 = Conv2D(32, 3, padding='same', data_format='channels_first')(net1)
    net1 = Conv2D(32, 3, padding='same', data_format='channels_first')(net1)
    net1 = MaxPooling2D(pool_size=(2, 2))(net1)
    net1 = Conv2D(64, 3, padding='same', data_format='channels_first')(net1)
    net1 = Conv2D(64, 3, padding='same', data_format='channels_first')(net1)
    net1 = MaxPooling2D(pool_size=(2, 2))(net1)
    net1 = Conv2D(128, 3, padding='same', data_format='channels_first')(net1)
    net1 = MaxPooling2D(pool_size=(2, 2))(net1)
    net1 = Flatten()(net1)

    in2 = Input(shape=(3, imsize, imsize))
    net2 = Conv2D(32, 3, padding='same', data_format='channels_first')(in2)
    net2 = Conv2D(32, 3, padding='same', data_format='channels_first')(net2)
    net2 = Conv2D(32, 3, padding='same', data_format='channels_first')(net2)
    net2 = Conv2D(32, 3, padding='same', data_format='channels_first')(net2)
    net2 = MaxPooling2D(pool_size=(2, 2))(net2)
    net2 = Conv2D(64, 3, padding='same', data_format='channels_first')(net2)
    net2 = Conv2D(64, 3, padding='same', data_format='channels_first')(net2)
    net2 = MaxPooling2D(pool_size=(2, 2))(net2)
    net2 = Conv2D(128, 3, padding='same', data_format='channels_first')(net2)
    net2 = MaxPooling2D(pool_size=(2, 2))(net2)
    net2 = Flatten()(net2)

    in3 = Input(shape=(3, imsize, imsize))
    net3 = Conv2D(32, 3, padding='same', data_format='channels_first')(in3)
    net3 = Conv2D(32, 3, padding='same', data_format='channels_first')(net3)
    net3 = Conv2D(32, 3, padding='same', data_format='channels_first')(net3)
    net3 = Conv2D(32, 3, padding='same', data_format='channels_first')(net3)
    net3 = MaxPooling2D(pool_size=(2, 2))(net3)
    net3 = Conv2D(64, 3, padding='same', data_format='channels_first')(net3)
    net3 = Conv2D(64, 3, padding='same', data_format='channels_first')(net3)
    net3 = MaxPooling2D(pool_size=(2, 2))(net3)
    net3 = Conv2D(128, 3, padding='same', data_format='channels_first')(net3)
    net3 = MaxPooling2D(pool_size=(2, 2))(net3)
    net3 = Flatten()(net3)

    #convpool = concatenate(convnets)
    convpool = concatenate([net1, net2, net3])
    convpool = Reshape((n_timewin, -1))(convpool)

    conv_out = Permute((2,1))(convpool)
    conv_out = Conv1D(64, 3)(conv_out)
    conv_out = Flatten()(conv_out)

    lstm_out = LSTM(128, activation='tanh')(convpool)

    dense_input = concatenate([lstm_out, conv_out])

    main_output = Dropout(0.5)(dense_input)
    main_output = Dense(512, activation='relu')(main_output)
    main_output = Dense(num_classes, activation='softmax')(main_output)

    lstm_conv = Model(inputs=[in1, in2, in3], outputs=main_output)

    #plot_model(lstm_conv, 'mix.png')

    opt = adam(clipvalue=100)

    lstm_conv.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)
    num_epochs = 5

    print('Training the LSTM-CONV Model...')

    X_train = [X_train[0], X_train[1], X_train[2]]

    lstm_conv.fit(X_train, y_train_cat, epochs=num_epochs)

    print('Done!')
