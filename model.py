import numpy as np 
from matplotlib import pyplot as plt 
from qr_decomposition import qr_decomposition

data = np.array([
    [1, 81], [1, 96], [1, 84], [1, 84], [1, 84], [1, 93], [1, 85], [1, 87],
    [2, 66], [2, 85], [2, 73], [2, 72], [2, 76], [2, 84], [2, 76], [2, 80],
    [3, 52], [3, 69], [3, 62], [3, 57], [3, 65], [3, 69], [3, 66], [3, 68],
    [4, 38], [4, 50], [4, 47], [4, 43], [4, 52], [4, 50], [4, 53], [4, 53]
])

m, n = data.shape
A = np.array([data[:,0], np.ones(m)]).T
b = data[:, 1] 

Q, R = qr_decomposition.givens_rotation(A)
b_hat = Q.T.dot(b) 

R_upper = R[:n, :]
b_upper = b_hat[:n]

x = np.linalg.solve(R_upper, b_upper) 
slope, intercept = x 

plt.scatter(data[:, 0], data[:, 1]) 
plt.title('Score / Grade')
plt.xlabel('Grade')
plt.ylabel('Score')
plt.axis([1, 4, 30, 100])
plt.plot(x, x*slope+intercept)
plt.show() 
