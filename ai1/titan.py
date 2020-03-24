from sklearn.datasets import load_iris
#导入IRIS数据集
iris = load_iris()

#print(iris)

#标准化，返回值为标准化后的数据
from sklearn.preprocessing import StandardScaler
irisstand =StandardScaler().fit_transform(iris.data)
print(irisstand)

#区间缩放，返回值为缩放到[0, 1]区间的数据
from sklearn.preprocessing import MinMaxScaler
iris01 = MinMaxScaler().fit_transform(iris.data)
print(iris01)
#归一化，返回值为归一化后的数据
from sklearn.preprocessing import Normalizer
irisnormal =Normalizer().fit_transform(iris.data)
print(irisnormal)