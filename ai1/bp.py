import numpy as np

class bp:
    def __init__(self):
        '''
                inputSize: X大小 3*2
                outputSize: Y大小 3*1
                hiddenSize: 隐层大小
                alpha: 学习率
                iterations: 训练循环次数
                count_error: 统计学习误差
                '''
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.alpha = 0.2
        self.iterations = 100
        self.count_error = []

        # 权重
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

        self.a2 = None
        self.a3 = None
        self.B1 = None
        self.B2 = None
        self.error = None
        self.delta3 = None
        self.delta2 = None


    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def forward(self, X, Y):
        self.a2 = self.sigmoid(np.dot(X, self.W1))
        self.a3 = self.sigmoid(np.dot(self.a2, self.W2))

        return self.a3

    def sigmoidDerivative(self, s):
        return s*(1-s)


    def backward(self, X, y, a3):
        self.error = self.a3 -y
        self.delta3 = self.error*self.sigmoidDerivative(a3)
        self.delta2 = self.delta3.dot(self.W2.T)*self.sigmoidDerivative(self.a2)

        self.W1 -= self.alpha*X.T.dot(self.delta2)
        self.W2 -= self.alpha*self.a2.T.dot(self.delta3)






