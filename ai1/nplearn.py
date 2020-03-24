import numpy as np
from io import BytesIO
x = np.float32(1.0)


data = "1, 2, 3\n4, 5, 6".encode()
ddd =np.genfromtxt(BytesIO(data), delimiter=",")

print(ddd)

x = np.arange(10,1,-1)



