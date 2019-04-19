import numpy as np
import matplotlib.pyplot as plt
knot = np.array([0,0,0,0.5,1,1,1])
n = knot.shape[0]

t = np.array(range(0, 1, 0.01))
tn = t.shape[0]
# n0 = np.array(np.zeros, np.float, )
n0 = np.zeros(tn, n)
for ti in t:
    for ki in range(n):
        
        n0[ti][ki] = 1

plt.plot(knot, '.')
plt.show()