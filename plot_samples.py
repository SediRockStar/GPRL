from datetime import datetime
from os import mkdir

import gym
import matplotlib.pyplot as plt
import numpy as np


from gym import envs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, RationalQuadratic, ExpSineSquared

from GPRL_sklearn import GPRL

T = 50
gamma = 0.9
draw = False
kernel = Matern(length_scale=2, nu=3 / 2)+ ConstantKernel(constant_value=2) + WhiteKernel(noise_level=1)



env = gym.make('MountainCar-v0')
env.reset()

gprl = GPRL(env,gamma= gamma, draw= draw)

gp = GaussianProcessRegressor(kernel=kernel, optimizer= "fmin_l_bfgs_b", n_restarts_optimizer=0, random_state= 2002)

gprl.GP_V = gp

gprl.run(T=T)

points= 21
X1 = np.linspace(-1.2, 0.6, points).reshape(-1, 1)


new_points = np.zeros((points, 2))  # create an empty array to store the new points
for i in range(points):
    new_points[i][0]= X1[i][0]
    new_points[i][1]= 0

y_mean, y_std = gp.predict(new_points, return_std=True)
y_mean= np.squeeze(y_mean)
y_std= np.squeeze(y_std)
# Generate samples using the GP model
y_samples = gp.sample_y(new_points, 3)
# Plot the samples
for i in range(3):
    plt.plot(np.squeeze(new_points[:, 0]), np.squeeze(np.squeeze(y_samples)[:, i]), label=f'Sample {i+1}', linestyle="--",)
plt.scatter(np.squeeze(new_points[:, 0]), np.squeeze(y_mean), label='Observations', color='red')
plt.fill_between(X1.ravel(), y_mean - y_std, y_mean + y_std, alpha=0.2, color='grey', label='Standard deviation')
plt.legend()
plt.show()