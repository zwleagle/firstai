
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import *
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.param ={}
        self.param['W1'] = weight_init_std*np.random.rand(input_size, hidden_size)
        self.param['b1'] = np.zeros(hidden_size)
        self.param['W2'] = weight_init_std*np.random.rand(hidden_size, output_size)
        self.param['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.param['W1'], self.param['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.param['W2'], self.param['b2'])

        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.sum(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t)/float(x.shape[0])

        return accuracy

   # 微分法求梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W:self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.param['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.param['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.param['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.param['b2'])

        return grads

    #反向传播法求梯度
    def gradient(self, x, t):
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db


        return grads



