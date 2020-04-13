
import numpy as np
import matplotlib.pyplot as plt


# 目标函数:y=x^2
def func(x):
    return np.square(x)


# 目标函数一阶导数:dy/dx=2*x
def dfunc(x):
    return 2 * x


def GD_decay(x_start, df, epochs, lr,  decay):
    """
        带有学习率衰减因子的梯度下降法。
        :param x_start: x的起始点
        :param df: 目标函数的一阶导函数
        :param epochs: 迭代周期
        :param lr: 学习率
        :param decay: 学习率衰减因子
        :return: x在每次迭代后的位置（包括起始点），长度为epochs+1
     """

    xs = np.zeros(epochs +1)
    x = x_start
    xs[0] = x
    v = 0

    for i in range(epochs):
        dx = df(x)
        lr_i = lr * 1.0/(1.0 + decay*i)
        v = -dx*lr_i
        x += v
        xs[i+1] = x

    return  xs

def demo3_GD_decay():
    line_x = np.linspace(-5, 5, 100)
    line_y = func(line_x)
    plt.figure('Gradient Desent: Decay')

    x_start = -5
    epochs = 10

    lr = [0.1, 0.3, 0.9, 0.99]
    decay = [0.0, 0.01, 0.5, 0.9]

    color = ['k', 'r', 'g', 'y']

    row = len(lr)
    col = len(decay)
    size = np.ones(epochs + 1) * 10
    size[-1] = 30
    for i in range(row):
        for j in range(col):
            x = GD_decay(x_start, dfunc, epochs, lr=lr[i], decay=decay[j])
            plt.subplot(row, col, i * col + j + 1)
            plt.plot(line_x, line_y, c='b')
            plt.plot(x, func(x), c=color[i], label='lr={}, de={}'.format(lr[i], decay[j]))
            plt.scatter(x, func(x), c=color[i], s=size)
            plt.legend(loc=0)
            #plt.show()
    plt.show()

demo3_GD_decay()