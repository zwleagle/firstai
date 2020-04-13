import numpy as np



def sigmod(x):
    return 1/(1+np.exp(-x))

def test(x):
    l1 = sigmod(np.dot(x,syn0))
    return sigmod(np.dot(l1,syn1))

alpha = 1 # 步长

X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],
              [1,0,0], [1,0,1], [1,1,0], [1,1,1]])

# 如果三个特征值取相同值，则分类为1，否则为0
y = np.array([[1,0,0,0,0,0,0,1]]).T

np.random.seed(1)

syn0 = 2 * np.random.random((3,11)) - 1 # 初始值为-1到1之间的数，当然也可以全部为0
syn1 = 2 * np.random.random((11,1)) - 1

for j in range(60000):
    l0 = X  # 输入层

    l1 = sigmod(np.dot(l0, syn0))  # 隐含层

    l2 = sigmod(np.dot(l1, syn1))  # 输出层

    l2_error = l2 - y  # 损失函数

    l2_delta = l2_error * l2 * (1 - l2)
    l1_delta = l2_delta.dot(syn1.T) * l1 * (
                1 - l1)  # 展开则为：l1_delta = l2_error * l2 * (1 - l2).dot(syn1.T) * l1 * (1 - l1)

    # Backpropagation
    syn1 += (-1) * alpha * l1.T.dot(l2_delta)
    syn0 += (-1) * alpha * l0.T.dot(l1_delta)

for x in X:

    print(test(x))  # 神经元权重矩阵已经计算完毕，可以用它来做一次验证


print(test([-1,-1, -1]) )
print(test([2,2,2]) )
print(test([3,3,3]))
