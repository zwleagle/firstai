import numpy as np
from ch05.two_layer_net import TwoLayerNet, softmax

X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],
              [1,0,0], [1,0,1], [1,1,0], [1,1,1]])

y = np.array([[1,0,0,0,0,0,0,1]]).T

network = TwoLayerNet(input_size=3, hidden_size=4, output_size=2)

iters_num = 500
train_size = X.shape[0]
batch_size = 1
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = X[batch_mask]
    t_batch = y[batch_mask]

    # 梯度
    #    grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_batch, t_batch)

        train_acc_list.append(train_acc)

        print(train_acc, i)

result = network.predict(np.array([[0,0,0]]))
print(softmax(result))