import pandas as pd
import numpy as np

# email data not standardized

# read data
email0 = pd.read_csv("spambase/spambase.data", header=None)
d0 = {'0': np.ones(4601)}
df0 = pd.DataFrame(d0)
email = pd.concat([df0, email0], axis=1, ignore_index=True)

# split training and testing data
n, p = email.shape
spam = email.loc[email[p - 1] == 1]
ham = email.loc[email[p - 1] == 0]
idx_spam, idx_ham = [np.arange(len(x)) < len(x) * 2 // 3 for x in (spam, ham)]

train_data = pd.concat([spam.loc[idx_spam], ham.loc[idx_ham]], ignore_index=True)
n1 = train_data.shape[0]
test_data = pd.concat([spam.loc[~idx_spam], ham.loc[~idx_ham]], ignore_index=True)

# gradient descent training
theta = np.zeros(p-1)
prob = np.zeros(n1)
tau = 1e2
threshold = 1e-3
iter_max = int(1e5)
x_train = train_data.drop(p-1, axis=1)
y_train = train_data[p-1]

for j in range(iter_max):
    prob = (1 + np.exp(- x_train @ theta)) ** (-1)
    grad = np.transpose(x_train) @ (prob - y_train) * (1 / n1)
    theta0 = theta.copy()
    theta += - tau * grad
    norm_diff = np.linalg.norm(theta - theta0) / (np.linalg.norm(theta0) + 1e-2)
    print(norm_diff, "\n")
    if norm_diff < threshold:
        print("Successfully converge. \n")
        break

# error rate for training set
prob_train = (1 + np.exp(- x_train @ theta)) ** (-1)
y_hat_train = round(prob_train)
err_train = np.mean(abs(y_hat_train - y_train))
print("error rate for training set = ", err_train, "\n")

# error rate for testing set
x_test = test_data.drop(p-1, axis=1)
y_test = test_data[p-1]
prob_test = (1 + np.exp(- x_test @ theta)) ** (-1)
y_hat_test = round(prob_test)
err_test = np.mean(abs(y_hat_test - y_test))
print("error rate for testing set = ", err_test, "\n")



