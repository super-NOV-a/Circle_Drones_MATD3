import numpy as np
from pprint import pprint

x = np.array([0, 0, 1, 0, 1])
y = np.array([[0, 0, 1, 0, 1]])
z = np.array([[[0, 0, 1, 0, 1]]])
idx = np.argwhere(x)
idy = np.argwhere(y)
idz = np.argwhere(z)
pprint(idx)
pprint(idy)
pprint(idz)
print(idz[0][0][2])
