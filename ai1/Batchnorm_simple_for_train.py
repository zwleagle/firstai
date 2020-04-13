
import  numpy as np

def Batchnorm_simple_for_train(x, gamma, beta, bn_param, eps):
    """
    param:x    : 输入数据，设shape(B,L)
    param:gama : 缩放因子  γ
    param:beta : 平移因子  β
    param:bn_param   : batchnorm所需要的一些参数
        eps      : 接近0的数，防止分母出现0
        momentum : 动量参数，一般为0.9， 0.99， 0.999
        running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
        running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
    """

    running_mean = bn_param['running_mean']  # shape = [B]
    running_var = bn_param['running_var']  # shape = [B]
    results = 0.  # 建立一个新的变量

    x_mean = x.mean(axis=0)  # 计算x的均值
    x_var = x.var(axis=0)  # 计算方差
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)  # 归一化
    results = gamma * x_normalized + beta

    running_mean = momentum * running_mean + (1 - momentum) * x_mean

