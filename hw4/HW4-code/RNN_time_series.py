from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.utils import plot_model

# read data
data = np.load("data.npy")


# visualizing data
plt.figure(figsize=(12, 6))
for i in range(48):
    plt.subplot(6, 8, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.plot(data[i, range(50)])
plt.show()
plt.savefig('train_examples2.png')

# train & testing data
x_train = data[:66, :499]
y_train = data[:66, 1:]
x_test = data[66:, :499]
y_test = data[66:, 1:]
x_train = x_train.reshape([-1, 499, 1])
y_train = y_train.reshape([-1, 499, 1])
x_test = x_test.reshape([-1, 499, 1])
y_test = y_test.reshape([-1, 499, 1])

# hyper-parameters
input_shape = (499, 1)
output_shape = 1
epochs = 5000


# model structure
model = Sequential()
# RNN layers
model.add(SimpleRNN(units=20,
                    dropout=0.2,
                    return_sequences=True,
                    input_shape=input_shape))
model.add(SimpleRNN(units=10,
                    dropout=0.2,
                    return_sequences=True))
model.add(SimpleRNN(units=5,
                    dropout=0.2,
                    return_sequences=True))
# full connected layers
model.add(Dense(output_shape))
model.summary()
plot_model(model, show_shapes=True, to_file='rnn_model.png')

# loss function
model.compile(loss='mean_squared_error',
              optimizer='adam')


# train the network
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    validation_data=(x_test, y_test))


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('loss.png')

# true vs estimate curves
y_hat = model.predict(x_test)


plt.figure(figsize=(16, 8))
for i in range(6):
    plt.subplot(3, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.plot(y_test[i, :])
    plt.plot(y_hat[i, :])
plt.show()
plt.savefig('fit.png')
