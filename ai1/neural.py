import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

c = mean_squared_error(np.array(y), np.array(t))
print(c)


# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


print(cross_entropy_error(np.array(y), np.array(t)))

import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    print(y)
    print(t)
    print(np.arange(batch_size))
    print(y[np.arange(batch_size), t])
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


print(cross_entropy_error(np.array(y), np.array(t)))

t1 = [1, 3]
y = np.array([[0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0.2, 0.8, 0, 0, 0, 0, 0, 0]])
batch_size1 = y.shape[0]
print(y[np.arange(batch_size1), t1])


# y=np.array([[0,1,0.1,0,0,0,0,0,0,0],[0,0,0.2,0.8,0,0,0,0,0,0]])
# t_onehot=np.array([[0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0]])#one-hot
#
# t = t_onehot.argmax(axis=1)#非one-hot
# print(t)
# batch_size = y.shape[0]
# print(batch_size)#2
# k=y[np.arange(batch_size), t] # [y[0,1] y[1,3]]
# print(k)
# print(k)#[1.  0.8]
# r=-np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))/ batch_size
# print(r)#0.11157166315711126

def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2.0


def function_tmp2(x1):
    return 3.0 ** 2.0 + x1 * x1


print(numerical_diff(function_tmp1, 3.0))


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值

    return grad


print(numerical_gradient(function_2, np.array([3.0, 4.0])))


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def function_2x(x):
    return x[0]**2 + (x[1])**2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
