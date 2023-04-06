# Testing extracting diagonal from list of P matricies

import numpy as np

N     = 4
P_est = np.zeros([N, 6, 6])  # state covariance matrices

P_est[0] = np.diag([1, 1, 1, 1, 1, 1])
P_est[1] = np.diag([2, 2, 3, 2, 2, 2])
P_est[2] = np.diag([3, 3, 3, 3, 3, 3])
P_est[3] = np.diag([4, 4, 4, 4, 4, 4])
P = P_est.diagonal(0,1,2)

print(P.shape)
print(P)

x_est = np.zeros([N, 6]) 
print(x_est)
