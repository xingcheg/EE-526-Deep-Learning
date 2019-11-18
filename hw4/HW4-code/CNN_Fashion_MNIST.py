from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model


# read data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# visualizing data
plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
plt.savefig("train_examples.png")

# pre-processing data
train_images = np.reshape(train_images, [-1, 28, 28, 1])
test_images = np.reshape(test_images, [-1, 28, 28, 1])
train_images = train_images / 255.0
test_images = test_images / 255.0


# hyper-parameters
num_labels = 10
input_shape = (28, 28, 1)
kernel_size = 3
pool_size = 2

batch_size = 128
epochs = 20


# convert class vectors to one hot matrix
train_labels = keras.utils.to_categorical(train_labels, num_labels)
test_labels = keras.utils.to_categorical(test_labels, num_labels)


# model structure
model = Sequential()
# convolutional layers
model.add(Conv2D(filters=64,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=64,
                 kernel_size=kernel_size,
                 activation='relu'))
# full connected layers
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
plot_model(model, show_shapes=True, to_file='cnn_model.png')
# loss function
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# train the network
history = model.fit(train_images, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(test_images, test_labels))


loss_train, acc_train = model.evaluate(train_images, train_labels, batch_size=batch_size)
print("\nTrain accuracy: %.1f%%" % (100.0 * acc_train))

loss_test, acc_test = model.evaluate(test_images, test_labels, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc_test))

# Train accuracy: 97.7%
# Test accuracy: 92.5%


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig("acc.png")
