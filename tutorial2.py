#a tutorial by julien simon

import mxnet as mx
a = mx.symbol.Variable('A')
b = mx.symbol.Variable('B')
c = mx.symbol.Variable('C')
d = mx.symbol.Variable('D')
e = (a*b)+(c*d) #to calculate e without knowing the values of a, b, c and d

import numpy as np
a_data = mx.nd.array([1], dtype=np.int32)
b_data = mx.nd.array([2], dtype=np.int32)
c_data = mx.nd.array([3], dtype=np.int32)
d_data = mx.nd.array([4], dtype=np.int32)

 executor=e.bind(mx.cpu(), {'A':a_data, 'B':b_data, 'C':c_data, 'D':d_data})
 e_data = executor.forward()
 e_data[0].asnumpy()
 
a_data = mx.nd.uniform(low=0, high=1, shape=(1000,1000))
b_data = mx.nd.uniform(low=0, high=1, shape=(1000,1000))
c_data = mx.nd.uniform(low=0, high=1, shape=(1000,1000))
d_data = mx.nd.uniform(low=0, high=1, shape=(1000,1000))
executor=e.bind(mx.cpu(), {'A':a_data, 'B':b_data, 'C':c_data, 'D':d_data})
e_data = executor.forward()

 e_data[0].asnumpy()
