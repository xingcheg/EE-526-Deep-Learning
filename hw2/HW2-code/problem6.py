import tensorflow as tf
from DNN import *


# normalize rows of X so that each row has mean zero and variance 1
def normalize(X):
    m = np.mean(X, axis=1, keepdims=True)
    var = np.var(X, axis=1, keepdims=True)
    var[var < 1E-5] = 1
    X = X - m
    X = X / np.sqrt(var)
    return X


mnist = tf.keras.datasets.mnist
(x, y), (x_test, y_test) = mnist.load_data()
X = x.transpose((1, 2, 0)).reshape(784, -1)
X = normalize(X)
X_test = x_test.transpose((1, 2, 0)).reshape(784, -1)
X_test = normalize(X_test)


# function to transform y
def y_tran10(y):
    y_mat = np.zeros([10, len(y)])
    for i in range(10):
        y_mat[i, np.where(y == i)] = 1
    return y_mat


def y_inv_tran(y_mat):
    y = np.argmax(y_mat, axis=0)
    return y


# NN train & test
def nn_mnist(x_train, y_train, x_test, y_test, layers, eta=0.1, niter=1000, batch_size=128):
    dim_in = x_train.shape[0]  # Input dimension
    nn = NeuralNetwork(dim_in, layers)
    nn.setRandomWeights(0.1)
    CE = ObjectiveFunction('crossEntropyLogit')
    y_train1 = y_tran10(y_train)

    for i in range(niter):   # train NN
        rand_idx = [np.random.randint(x_train.shape[1]) for j in range(batch_size)]
        x_train_sub = x_train[:, rand_idx]
        y_train_sub = y_train1[:, rand_idx]
        logp = nn.doForward(x_train_sub)
        J = CE.doForward(logp, y_train_sub)
        dz = CE.doBackward(y_train_sub)
        nn.doBackward(dz)
        nn.updateWeights(eta)
        if i % 1000 == 0:  # train & test error
            y_train_hat = y_inv_tran(nn.predict(x_train))
            y_test_hat = y_inv_tran(nn.predict(x_test))
            train_err = sum(y_train_hat != y_train)/len(y_train)
            test_err = sum(y_test_hat != y_test)/len(y_test)
            print("iter = ", i, ";\ttrain error = ", train_err, "\ttest error = ", test_err, "\t loss = ", J, "\n")


# nn_mnist(X, y, X_test, y_test, layers=[(10, Linear)], eta=0.1, niter=20001, batch_size=256)
# iter =  20000 ;	train error =  0.061266666666666664 	test error =  0.0752 	 loss =  0.139948771673014

# nn_mnist(X, y, X_test, y_test, layers=[(50, ReLU),  (10, Linear)], eta=0.25, niter=20001, batch_size=256)
# iter =  20000 ;	train error =  3.3333333333333335e-05 	test error =  0.0303 	 loss =  0.000937989163990402

# nn_mnist(X, y, X_test, y_test, layers=[(100, ReLU),  (10, Linear)], eta=0.25, niter=20001, batch_size=256)
# iter =  20000 ;	train error =  0.0 	test error =  0.0261 	 loss =  0.0006825340884620424

# nn_mnist(X, y, X_test, y_test, layers=[(140, ReLU),  (10, Linear)], eta=0.25, niter=20001, batch_size=256)
# iter =  20000 ;	train error =  0.0 	test error =  0.0293 	 loss =  0.0006356000582702856

# nn_mnist(X, y, X_test, y_test, layers=[(50, ReLU), (50, ReLU), (10, Linear)], eta=0.1, niter=20001, batch_size=256)
# iter =  20000 ;	train error =  0.0013333333333333333 	test error =  0.0384 	 loss =  0.001822203193588888

# nn_mnist(X, y, X_test, y_test, layers=[(100, ReLU), (20, ReLU), (10, Linear)], eta=0.1, niter=20001, batch_size=256)
# iter =  20000 ;	train error =  0.0023 	test error =  0.0405 	 loss =  0.01145676953506744



