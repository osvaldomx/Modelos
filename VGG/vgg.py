import numpy as np

from keras import applications
# from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from utils import reformatInput


# dimensions of our images.
images = np.load('imagenes_32.npy')
labels = np.load('labels_32.npy')
print labels.shape
fold_pairs = np.load('fold_pairs_32.npy')
fold = fold_pairs[2]

top_model_weights_path = 'bottleneck_fc_model.h5'
epochs = 10
batch_size = 32
#input_tensor = Input(shape=(32,32,3))
input_tensor = Input(shape=(32,32,3))

num_classes = len(np.unique(labels))
(X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)
X_train = X_train.astype("float32", casting='unsafe')
X_val = X_val.astype("float32", casting='unsafe')
X_test = X_test.astype("float32", casting='unsafe')


images = np.swapaxes(images,0,2)
X_train = np.swapaxes(X_train,1,3)
X_val = np.swapaxes(X_val,1,3)
X_test = np.swapaxes(X_test,1,3)


# build the VGG16 network
base_model = applications.VGG16(weights=None,
                                include_top=False,
                                input_tensor=input_tensor)

print('Base Model loaded...')

# build a classifier model to puto on top of the
# convolutional model

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

print('Top Model builded...')
# note that it is necessary to start with a 
# fully-trained classifier, including the top 
# classifier, in order to successfully do fine-tuning
#top_model.load_weights(top_model_weights_path)
print('Weights loaded...')

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])

# fine-tune the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

#X_all = np.concatenate((X_train, X_val, X_test))
#y_all = np.concatenate((y_train, y_val, y_test))

#model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)

print('Done!')


