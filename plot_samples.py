from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, RationalQuadratic, ExpSineSquared
from GPRL_sklearn import GPRL
import numpy as np

X1 = np.linspace(-1.2, 0.6, 200).reshape(-1, 1)

new_points = np.zeros((100, 2))  # create an empty array to store the new points
for i in range(100):
    new_points[i][0]= X1[i][0]
    new_points[i][1]= 0

# Generate samples using the GP model
y_samples = gp.sample_y(new_points, 5)
# Plot the samples
for i in range(5):
    plt.plot(np.squeeze(new_points[:, 0]), np.squeeze(np.squeeze(y_samples)[:, i]), label=f'Sample {i+1}')
plt.show()