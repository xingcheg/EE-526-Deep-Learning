from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


# read data
data = np.load("data.npy")


# train & testing data
x_train = data[:66, :495]
y_train = data[:66, 5:]
x_test = data[66:, :495]
y_test = data[66:, 5:]
x_train = x_train.reshape([-1, 495, 1])
y_train = y_train.reshape([-1, 495, 1])
x_test = x_test.reshape([-1, 495, 1])
y_test = y_test.reshape([-1, 495, 1])

# hyper-parameters
input_shape = (495, 1)
output_shape = 1
batch_size = 33
epochs = 4000


# model structure
model = Sequential()
# RNN layers
model.add(SimpleRNN(units=32,
                    dropout=0.3,
                    return_sequences=True,
                    input_shape=input_shape))
model.add(SimpleRNN(units=32,
                    dropout=0.3,
                    return_sequences=True))
model.add(SimpleRNN(units=64,
                    dropout=0.3,
                    return_sequences=True))
# full connected layers
model.add(Dense(output_shape))
model.summary()

# loss function
model.compile(loss='mean_squared_error',
              optimizer='adam')


# train the network
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test))


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('loss_pred5.png')

# true vs estimate curves
y_hat = model.predict(x_test)


plt.figure(figsize=(16, 8))
for i in range(34):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.plot(y_test[i, 470:])
    plt.plot(y_hat[i, 470:])
plt.show()
plt.savefig('fit_pred5.png')

