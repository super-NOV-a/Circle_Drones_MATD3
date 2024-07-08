import itertools

import numpy as np

ret = 0
for i, j in itertools.combinations(range(3), 2):
    print("each other", i, j)
    print("target", i)

