from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy
import math
import sys

import six

BATCH_SIZE = 20

filename = "/Users/zouwanli/PycharmProjects/housing.data";

feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'convert'
]
feature_num = len(feature_names)
data = numpy.fromfile(filename, sep=' ') # 从文件中读取原始数据
data = data.reshape(data.shape[0] // feature_num, feature_num)
maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]

for i in six.moves.range(feature_num-1):
   data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i]) # six.moves可以兼容python2和python3

ratio = 0.8 # 训练集和验证集的划分比例
offset = int(data.shape[0]*ratio)
train_data = data[:offset]
test_data = data[offset:]

def reader_creator(train_data):
    def reader():
        for d in train_data:
            yield d[:-1], d[-1:]
    return reader

train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader_creator(train_data), buf_size=500),
        batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        reader_creator(test_data), buf_size=500),
        batch_size=BATCH_SIZE)


x = fluid.layers.data(name='x', shape=[13], dtype='float32') # 定义输入的形状和数据类型
y = fluid.layers.data(name='y', shape=[1], dtype='float32') # 定义输出的形状和数据类型
y_predict = fluid.layers.fc(input=x, size=1, act=None) # 连接输入和输出的全连接层

main_program = fluid.default_main_program() # 获取默认/全局主函数
startup_program = fluid.default_startup_program() # 获取默认/全局启动程序

cost = fluid.layers.square_error_cost(input=y_predict, label=y) # 利用标签数据和输出的预测数据估计方差
avg_loss = fluid.layers.mean(cost) # 对方差求均值，得到平均损失


#克隆main_program得到test_program
#有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
#该api不会删除任何操作符,请在backward和optimization之前使用
test_program = main_program.clone(for_test=True)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)


use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace() # 指明executor的执行场所

###executor可以接受传入的program，并根据feed map(输入映射表)和fetch list(结果获取表)向program中添加数据输入算子和结果获取算子。使用close()关闭该executor，调用run(...)执行program。
exe = fluid.Executor(place)



num_epochs = 100

def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program,
                            feed=feeder.feed(data_test),
                            fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)] # 累加测试过程中的损失值
        count += 1 # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated] # 计算平均损失



#%matplotlib inline
params_dirname = "fit_a_line.inference.model"
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe.run(startup_program)
train_prompt = "train cost"
test_prompt = "test cost"
from paddle.utils.plot import Ploter
plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

exe_test = fluid.Executor(place)

for pass_id in range(num_epochs):
    for data_train in train_reader():
        avg_loss_value, = exe.run(main_program,
                                  feed=feeder.feed(data_train),
                                  fetch_list=[avg_loss])
        if step % 10 == 0: # 每10个批次记录并输出一下训练损失
            plot_prompt.append(train_prompt, step, avg_loss_value[0])
            plot_prompt.plot()
            print("%s, Step %d, Cost %f" %
	                  (train_prompt, step, avg_loss_value[0]))
        if step % 100 == 0:  # 每100批次记录并输出一下测试损失
            test_metics = train_test(executor=exe_test,
                                     program=test_program,
                                     reader=test_reader,
                                     fetch_list=[avg_loss.name],
                                     feeder=feeder)
            plot_prompt.append(test_prompt, step, test_metics[0])
            plot_prompt.plot()
            print("%s, Step %d, Cost %f" %
	                  (test_prompt, step, test_metics[0]))
            if test_metics[0] < 10.0: # 如果准确率达到要求，则停止训练
                break

        step += 1

        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")

        #保存训练参数到之前给定的路径中
        if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)
