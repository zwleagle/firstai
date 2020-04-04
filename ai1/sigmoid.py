
import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None


    def forward(self, x):
        out = 1/(1+ np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*(1.0 -self.out) * self.out

        return dx


n = 5
p =3
m = 2

X = np.random.randn(n, p)
W = np.random.randn(p, m)
b = np.zeros(m)
print(X)
print(W)
print(b)

Y = np.dot(X, W) + b

print(Y)