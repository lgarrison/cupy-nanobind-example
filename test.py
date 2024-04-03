import cupy as cp

import example.cuda

c = cp.arange(10., dtype=cp.float32)
d = cp.empty_like(c)

example.cuda.double_arr(d, c)
print(d)
