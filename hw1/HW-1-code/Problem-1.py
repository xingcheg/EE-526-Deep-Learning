import numpy as np
import matplotlib.pyplot as plt

# (a)
x0 = np.arange(0, 51, 1)
x = x0/50

# (b)
y0 = 1 + 2 * np.sin(5 * x) - np.sin(15 * x)
e = np.random.normal(0, 1, 51)
y = y0 + e


# (c)
X_t = np.array([x ** i for i in range(2)])
X = np.transpose(X_t)
y_hat = X @ np.linalg.inv(X_t @ X) @ X_t @ y

plt.figure()
plt.scatter(x, y)
plt.plot(x, y_hat, color='black')
plt.savefig("fig1_0.png")
plt.show()


# (d)
plt.figure()
for i in range(30):
    e = np.random.normal(0, 1, 51)
    y = y0 + e
    y_hat = X @ np.linalg.inv(X_t @ X) @ X_t @ y
    plt.plot(x, y_hat, color='black')
plt.scatter(x, y)
plt.plot(x, y0, color='red')
plt.savefig("fig9.png")
plt.show()


# (e)

# k = 3
X_t = np.array([x ** i for i in range(4)])
X = np.transpose(X_t)
plt.figure()
for i in range(30):
    e = np.random.normal(0, 1, 51)
    y = y0 + e
    y_hat = X @ np.linalg.inv(X_t @ X) @ X_t @ y
    plt.plot(x, y_hat, color='black')
plt.scatter(x, y)
plt.plot(x, y0, color='red')
plt.savefig("fig9.png")
plt.show()


# k = 5
X_t = np.array([x ** i for i in range(6)])
X = np.transpose(X_t)
plt.figure()
for i in range(30):
    e = np.random.normal(0, 1, 51)
    y = y0 + e
    y_hat = X @ np.linalg.inv(X_t @ X) @ X_t @ y
    plt.plot(x, y_hat, color='black')
plt.scatter(x, y)
plt.plot(x, y0, color='red')
plt.savefig("fig9.png")
plt.show()


# k = 7
X_t = np.array([x ** i for i in range(8)])
X = np.transpose(X_t)
plt.figure()
for i in range(30):
    e = np.random.normal(0, 1, 51)
    y = y0 + e
    y_hat = X @ np.linalg.inv(X_t @ X) @ X_t @ y
    plt.plot(x, y_hat, color='black')
plt.scatter(x, y)
plt.plot(x, y0, color='red')
plt.savefig("fig9.png")
plt.show()


# k = 9
X_t = np.array([x ** i for i in range(10)])
X = np.transpose(X_t)
plt.figure()
for i in range(30):
    e = np.random.normal(0, 1, 51)
    y = y0 + e
    y_hat = X @ np.linalg.inv(X_t @ X) @ X_t @ y
    plt.plot(x, y_hat, color='black')
plt.scatter(x, y)
plt.plot(x, y0, color='red')
plt.savefig("fig9.png")
plt.show()


# k = 11
X_t = np.array([x ** i for i in range(12)])
X = np.transpose(X_t)
plt.figure()
for i in range(30):
    e = np.random.normal(0, 1, 51)
    y = y0 + e
    y_hat = X @ np.linalg.inv(X_t @ X) @ X_t @ y
    plt.plot(x, y_hat, color='black')
plt.scatter(x, y)
plt.plot(x, y0, color='red')
plt.savefig("fig9.png")
plt.show()

