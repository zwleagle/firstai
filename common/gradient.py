# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad



#数值微分方法计算梯度
#个人对这段算法的理解：
#1.设置h值用来数值微分法的计算
#2.设置一个全为0的数组grad用来存放梯度，它和输入数据x的shape一样
#3.将x数组设置成可以修改并且可以进行多重索引（这样就可以用（a1,a2）的坐标形式定位哪一个x了）
#4.使用while循环遍历x数组，对每一个x进行计算数值微分，并将计算结果保存在梯度数组中

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    # 初始化数组为0用来存放梯度
    # 默认情况下，nditer将视待迭代遍历的数组为只读对象（read-only）
    # 为了在遍历数组的同时，实现对数组元素值得修改，必须指定op_flags=['readwrite']模式。
    # flags=['multi_index']表示对x进行多重索引

    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index  #把元素的索引（it.multi_index）赋值给idx
        # 用来还原值
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()
        # nditer.iternext()检查是否保留迭代，并在不返回结果的情况下执行单个内部迭代。
        # it.iternext()表示进入下一次迭代，如果不加这一句的话，输出的结果就一直都是(0, 0)。
        # 迭代（Iteration）如果给定一个list或tuple，我们可以通过for循环来遍历这个list或tuple
        # 迭代操作就是对于一个集合，无论该集合是有序还是无序，我们用 for 循环总是可以依次取出集合的每一个元素。

    return grad