
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)


x = np.random.randn(1000, 100 ) # 1000个数据
node_num = 100 # 各隐藏层的节点（神经元）数
hidden_layer_size = 5 # 隐藏层有5层

activations = {} # 激活值的结果保存在这里

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

   # w = np.random.randn(node_num, node_num) * 0.01
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    #w = np.random.randn(node_num, node_num) * 1
    z = np.dot(x, w)
    #a = sigmoid(z)
    a = tanh(z)
    activations[i] = a

activations[hidden_layer_size] = x

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 10, range=(0,1))
    print(a)
    b =a.flatten()
plt.show()