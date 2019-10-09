import pandas as pd
import datetime
# import numpy as np
from sklearn.preprocessing import StandardScaler
from DNN import *

# email data standardized

# read and standardize data
email0 = pd.read_csv("spambase/spambase.data", header=None)
n0, p0 = email0.shape
email0_x = email0.drop(p0-1, axis=1)
email0_y = email0[[p0-1]]
email0_scale = StandardScaler().fit(email0_x)
email0_x_std = email0_scale.transform(email0_x)
email0_x_std = pd.DataFrame(email0_x_std)

d0 = {'0': np.ones(n0)}
df0 = pd.DataFrame(d0)
email = pd.concat([df0, email0_x_std, email0_y], axis=1, ignore_index=True)

# split training and testing data
n, p = email.shape
spam = email.loc[email[p - 1] == 1]
ham = email.loc[email[p - 1] == 0]
idx_spam, idx_ham = [np.arange(len(x)) < len(x) * 2 // 3 for x in (spam, ham)]

train_data = pd.concat([spam.loc[idx_spam], ham.loc[idx_ham]], ignore_index=True)
n1 = train_data.shape[0]
test_data = pd.concat([spam.loc[~idx_spam], ham.loc[~idx_ham]], ignore_index=True)


# function to transform y
def y_tran(y):
    y_mat = np.zeros([2, len(y)])
    y_mat[0, np.where(y == 0)] = 1
    y_mat[1, np.where(y == 1)] = 1
    return y_mat


def y_inv_tran(y_mat):
    y = np.argmax(y_mat, axis=0)
    return y


# NN train & test

def nn_spam(x_train, y_train, x_test, y_test, eta=0.1, niter=10000):
    dim_in = x_train.shape[0]  # Input dimension
    dim_out = 2  # number of outputs
    layers = [(80, ReLU), (40, ReLU), (dim_out, Linear)]
    nn = NeuralNetwork(dim_in, layers)
    nn.setRandomWeights(0.1)
    CE = ObjectiveFunction('crossEntropyLogit')
    y_train1 = y_tran(y_train)

    t1 = datetime.datetime.now()
    for i in range(niter):   # train NN
        logp = nn.doForward(x_train)
        CE.doForward(logp, y_train1)
        dz = CE.doBackward(y_train1)
        nn.doBackward(dz)
        nn.updateWeights(eta)
        if (i % 100 == 0):  # train & test error
            y_train_hat = y_inv_tran(nn.predict(x_train))
            y_test_hat = y_inv_tran(nn.predict(x_test))
            train_err = np.mean(abs(y_train_hat - y_train))
            test_err = np.mean(abs(y_test_hat - y_test))
            print("iter = ", i, ";\ttrain error = ", train_err, "\ttest error = ", test_err, "\n")
    t2 = datetime.datetime.now()
    print("total time = ", t2 - t1, "\n")


x_train0 = train_data.drop(p-1, axis=1)
x_test0 = test_data.drop(p-1, axis=1)
x_train = np.transpose(x_train0)
x_test = np.transpose(x_test0)
y_train = train_data[p-1]
y_test = test_data[p-1]

nn_spam(x_train, y_train, x_test, y_test, eta=0.1)