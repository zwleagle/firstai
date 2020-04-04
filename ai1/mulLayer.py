
import numpy as np

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y

        return out

    def backward(self, dout):
        dx = dout*self.y
        dy = dout*self.x

        return dx, dy


apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)


dprice = 1

dapple_price, dtax = mul_tax_layer.backward(dprice)
print(dapple_price)
print(dtax)

dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple)
print(dapple_num)


y = np.array([1,2,3])
print(y.ndim)
print(y.size)
y.reshape(1, y.size)
print(y)
print(y.shape[0])
