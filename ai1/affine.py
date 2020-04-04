
import numpy as np


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None  # 用于计算反向传播时W的梯度
        self.dW  = None # 用于存储反向传播时计算出的W的梯度
        self.db = None  # 用于存储反向传播时计算出的b的梯度

    # 前向函数就是通过x,W和b计算出加权和，再输出
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    # 反向函数就是将上游传来的导数dout乘以权重矩阵的转置WT后输出
    # （反向函数的输出永远是上游传来的导数乘以该层正向输出对正向输入的偏导）
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout) # 顺便计算出上游导数关于该层权重和偏置的导数，实际上就是求出了损失函数关于该层权重和偏置的梯度
        self.db = np.sum(dout, axis=0)

        return dx

