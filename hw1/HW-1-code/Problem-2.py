import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([1, 1, 2, 4, 3, 2])
x2 = np.array([2, 4, 2, 2, 4, 3])
y = np.array([1, 1, 1, -1, -1, -1])
X = np.column_stack((np.ones(6), x1, x2))

theta = np.array([0., 0., 0.])
y_hat = np.array([0, 0, 0, 0, 0, 0])
eta = 0.1
iter_max = int(1e4)

for i in range(iter_max):
    for j in range(6):
        y_hat[j] = np.sign(X[j, :] @ theta)
        theta += eta * (y[j] - y_hat[j]) * X[j, :]
    if np.linalg.norm(y - y_hat, ord=0) == 0:
        break
print(theta)

# plot
z2 = np.array([min(x2)-2, max(x2)+2])
plt.figure()
plt.scatter(x1[y == 1], x2[y == 1])
plt.scatter(x1[y == -1], x2[y == -1])
plt.plot(-(theta[0] + theta[2] * z2)/theta[1], z2, color="black")
plt.savefig("perceptron.png")
plt.show()



