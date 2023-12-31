import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy.interpolate import make_interp_spline


'''
episodes = 40000 mode = NORMAL
hidden_dim = [300,300]
alpha = 0.1
tau = 1e-3
discount = 0.96
batch_size = 256
epsilon = 1e-4 
max_size = 1000000
'''

reward_f1       = [9, 9, 6, 15, 14, 12, 13, 13, 21, 16, 19, 21, 14, 22, 22, 20, 18, 22, 28, 12, 23, 12, 13, 28, 28, 22, 22, 22, 24, 24, 19, 26, 24, 21, 10, 27, 19, 29, 30, 35, 26, 30, 24, 19, 26, 36, 26, 34, 32, 31, 34, 24, 30, 41, 36, 31, 40, 27, 32, 36, 24, 27, 29, 40, 34, 42, 34, 40, 36, 38, 34, 32, 39, 40, 25, 28, 36, 40, 39, 33, 39, 33, 40, 37, 32, 33, 38, 32, 41, 33, 33, 39, 35, 38, 30, 33, 38, 40, 43, 37, 33, 29, 51, 36, 37, 36, 44, 37, 37, 48, 39, 38, 42, 41, 41, 44, 42, 42, 35, 40, 35, 37, 31, 28, 40, 33, 32, 38, 27, 38, 36, 38, 39, 38, 46, 35, 45, 33, 41, 35, 36, 51, 37, 34, 41, 41, 43, 42, 31, 49, 46, 49, 45, 38, 46, 44, 41, 39, 48, 52, 36, 42, 38, 46, 28, 40, 36, 31, 42, 34, 31, 36, 35, 43, 35, 48, 34, 42, 38, 43, 43, 45, 47, 35, 34, 46, 44, 40, 39, 43, 34, 32, 40, 50, 49, 48, 37, 40, 41, 47, 37, 38, 50, 42, 40, 37, 37, 43, 42, 43, 47, 49, 36, 43, 48, 45, 44, 45, 49, 50, 51, 50, 44, 43, 49, 36, 31, 34, 46, 58, 37, 49, 59, 34, 55, 39, 47, 55, 45, 55, 43, 44, 53, 54, 42, 47, 42, 48, 37, 36, 45, 47, 40, 40, 41, 39, 47, 52, 46, 40, 43, 43, 36, 44, 37, 43, 36, 42, 37, 35, 44, 43, 52, 48, 43, 47, 43, 50, 37, 45, 50, 44, 48, 53, 39, 50, 46, 38, 45, 51, 55, 50, 35, 48, 47, 46, 38, 39, 41, 41, 40, 44, 43, 40, 47, 40, 48, 45, 43, 44, 46, 43, 39, 47, 46, 49, 51, 50, 41, 57, 54, 43, 43, 49, 49, 46, 52, 48, 37, 48, 35, 44, 46, 49, 52, 46, 49, 45, 49, 51, 42, 45, 46, 53, 52, 44, 60, 45, 54, 53, 51, 43, 45, 47, 53, 49, 44, 42, 49, 42, 48, 51, 57, 56, 44, 58, 51, 53, 56, 55, 49, 52, 48, 53, 50, 49, 51, 50, 47, 40, 47, 55, 43, 48, 55, 50, 52, 46, 52, 51, 57, 41, 57, 50, 48, 55, 43, 41, 43, 48]
reward_f10      = [5, 13, 17, 9, 19, 28, 14, 20, 13, 12, 21, 11, 10, 21, 23, 24, 24, 26, 22, 15, 27, 23, 28, 23, 24, 24, 19, 25, 26, 25, 23, 29, 26, 27, 27, 32, 31, 25, 31, 28, 27, 34, 33, 30, 21, 29, 31, 32, 32, 33, 29, 22, 35, 26, 28, 39, 37, 31, 31, 37, 31, 32, 31, 35, 28, 30, 30, 31, 34, 37, 33, 37, 28, 37, 37, 43, 29, 37, 41, 31, 37, 42, 37, 40, 35, 37, 33, 29, 51, 40, 40, 41, 34, 39, 44, 45, 43, 39, 39, 32, 43, 48, 42, 39, 43, 40, 31, 31, 35, 38, 38, 37, 42, 45, 41, 35, 43, 42, 40, 39, 41, 36, 39, 40, 30, 40, 36, 45, 42, 40, 40, 43, 44, 47, 46, 49, 45, 45, 47, 34, 50, 43, 47, 44, 40, 51, 39, 41, 40, 46, 47, 48, 41, 36, 42, 42, 54, 51, 50, 44, 51, 56, 33, 37, 53, 42, 47, 43, 54, 47, 51, 38, 43, 46, 45, 51, 46, 42, 36, 50, 43, 42, 44, 40, 47, 49, 37, 45, 30, 42, 48, 49, 50, 47, 51, 41, 43, 41, 47, 44, 41, 40, 48, 39, 44, 49, 52, 48, 42, 45, 49, 42, 43, 37, 40, 48, 49, 41, 49, 37, 55, 55, 51, 49, 43, 46, 44, 50, 57, 41, 45, 51, 46, 41, 54, 50, 45, 52, 52, 41, 46, 42, 41, 49, 43, 45, 42, 51, 45, 51, 51, 45, 48, 47, 55, 56, 54, 41, 49, 45, 54, 47, 45, 40, 52, 42, 44, 44, 49, 43, 46, 52, 52, 47, 54, 47, 51, 49, 50, 51, 47, 53, 47, 50, 48, 45, 56, 42, 48, 47, 48, 40, 50, 46, 48, 52, 41, 47, 46, 49, 50, 47, 43, 44, 49, 41, 44, 49, 54, 48, 48, 44, 49, 49, 49, 49, 55, 53, 48, 48, 43, 57, 42, 54, 58, 54, 55, 52, 49, 36, 47, 45, 54, 44, 44, 52, 50, 38, 45, 44, 47, 44, 51, 37, 52, 44, 46, 45, 45, 37, 51, 53, 51, 46, 46, 47, 49, 49, 47, 49, 50, 45, 55, 51, 43, 46, 43, 47, 58, 41, 48, 52, 47, 53, 52, 41, 56, 45, 51, 50, 46, 44, 48, 51, 58, 53, 57, 43, 54, 49, 47, 55, 56, 47, 45, 50, 45, 58, 52, 51]
reward_f1_ft    = [4, 10, 16, 18, 14, 20, 16, 23, 25, 16, 30, 25, 26, 24, 20, 16, 22, 14, 27, 26, 29, 25, 18, 19, 21, 22, 30, 30, 32, 33, 26, 28, 36, 32, 35, 31, 39, 33, 41, 35, 37, 33, 28, 36, 39, 30, 31, 38, 47, 31, 31, 34, 38, 33, 27, 31, 38, 38, 38, 40, 38, 37, 39, 31, 30, 37, 27, 34, 37, 34, 39, 34, 29, 41, 39, 32, 33, 34, 37, 36, 36, 47, 47, 34, 32, 49, 46, 37, 40, 24, 38, 37, 42, 47, 44, 39, 44, 46, 44, 40, 40, 42, 41, 49, 46, 44, 39, 51, 40, 56, 50, 44, 37, 47, 42, 41, 34, 37, 42, 44, 37, 39, 49, 48, 45, 46, 50, 41, 50, 36, 39, 38, 39, 45, 43, 46, 43, 50, 38, 44, 44, 45, 44, 49, 50, 50, 45, 54, 40, 43, 43, 43, 62, 47, 55, 44, 54, 49, 47, 51, 44, 52, 58, 52, 49, 38, 49, 50, 50, 48, 45, 44, 44, 50, 49, 37, 46, 43, 56, 53, 48, 51, 37, 45, 55, 52, 54, 54, 51, 45, 46, 45, 60, 57, 45, 47, 56, 47, 44, 47, 51, 49, 44, 49, 45, 54, 45, 39, 51, 40, 35, 47, 49, 42, 53, 48, 44, 51, 43, 44, 56, 45, 42, 38, 45, 50, 45, 49, 47, 46, 48, 68, 37, 60, 57, 59, 55, 40, 52, 43, 51, 51, 53, 43, 57, 48, 49, 46, 52, 46, 55, 51, 50, 45, 47, 51, 49, 52, 48, 54, 46, 51, 51, 45, 49, 48, 48, 52, 57, 50, 58, 55, 51, 57, 51, 47, 50, 51, 43, 45, 40, 50, 47, 47, 51, 54, 49, 48, 46, 46, 58, 58, 40, 49, 50, 53, 52, 43, 44, 49, 49, 53, 47, 43, 42, 47, 53, 50, 50, 46, 56, 48, 53, 53, 48, 47, 45, 44, 38, 48, 42, 53, 38, 47, 48, 33, 40, 54, 46, 48, 50, 44, 52, 47, 43, 45, 46, 54, 47, 50, 54, 47, 54, 51, 46, 54, 48, 40, 48, 47, 50, 51, 50, 52, 44, 49, 52, 46, 55, 42, 53, 38, 45, 50, 55, 41, 45, 40, 50, 47, 48, 50, 50, 52, 43, 45, 50, 44, 50, 49, 47, 52, 46, 45, 52, 51, 52, 57, 50, 46, 46, 51, 48, 57, 50, 50, 41, 47, 49, 57]
reward_f5_ft    = [10, 17, 14, 9, 24, 11, 13, 18, 17, 17, 11, 19, 17, 20, 16, 24, 18, 16, 18, 24, 25, 21, 25, 20, 24, 24, 28, 31, 30, 27, 27, 28, 26, 30, 33, 25, 19, 30, 25, 28, 24, 30, 24, 24, 25, 33, 32, 27, 28, 27, 24, 32, 33, 39, 25, 21, 31, 30, 29, 32, 39, 35, 24, 35, 26, 29, 31, 26, 23, 29, 32, 32, 32, 38, 33, 39, 33, 33, 40, 31, 39, 39, 26, 33, 43, 35, 47, 41, 29, 48, 42, 42, 32, 33, 47, 32, 34, 37, 45, 42, 47, 45, 39, 40, 53, 36, 47, 45, 37, 50, 44, 51, 45, 48, 43, 31, 43, 45, 48, 42, 36, 43, 50, 41, 45, 56, 39, 40, 37, 52, 46, 36, 49, 40, 31, 45, 39, 46, 38, 46, 31, 42, 47, 51, 44, 43, 40, 51, 40, 42, 53, 50, 45, 47, 49, 44, 45, 37, 47, 42, 51, 51, 50, 44, 52, 37, 48, 61, 47, 46, 40, 42, 42, 48, 35, 47, 42, 51, 50, 47, 52, 43, 54, 57, 52, 42, 48, 49, 45, 48, 47, 54, 47, 44, 48, 49, 46, 39, 39, 48, 54, 50, 42, 49, 48, 43, 53, 52, 45, 49, 46, 52, 49, 50, 43, 42, 51, 43, 45, 49, 63, 48, 41, 47, 48, 51, 38, 47, 50, 55, 47, 43, 44, 42, 41, 53, 53, 47, 45, 51, 38, 45, 45, 50, 56, 53, 50, 55, 50, 52, 43, 40, 53, 53, 54, 47, 52, 39, 51, 50, 49, 55, 52, 55, 50, 47, 37, 46, 36, 45, 49, 50, 48, 51, 46, 52, 48, 50, 46, 48, 40, 50, 56, 44, 46, 41, 47, 51, 50, 51, 50, 48, 51, 42, 53, 44, 51, 52, 52, 46, 52, 48, 44, 56, 52, 52, 49, 52, 45, 51, 47, 45, 51, 50, 53, 47, 54, 50, 60, 45, 56, 51, 40, 42, 54, 44, 45, 54, 48, 46, 51, 54, 57, 51, 46, 47, 62, 57, 50, 55, 45, 44, 43, 50, 46, 48, 42, 45, 52, 42, 56, 48, 46, 51, 49, 45, 57, 54, 46, 51, 47, 50, 48, 49, 47, 51, 53, 42, 55, 48, 51, 46, 42, 39, 54, 47, 53, 47, 43, 43, 43, 53, 55, 57, 55, 47, 51, 49, 49, 53, 46, 45, 48, 56, 48, 51, 49, 44, 54, 53]
reward_f7_ft    = [5, 10, 14, 23, 16, 18, 14, 22, 13, 14, 17, 17, 18, 20, 16, 14, 24, 27, 20, 24, 18, 18, 19, 19, 25, 24, 17, 21, 22, 29, 33, 22, 30, 27, 18, 26, 28, 31, 33, 23, 40, 28, 27, 31, 33, 33, 36, 40, 25, 33, 36, 32, 32, 44, 41, 35, 29, 31, 41, 31, 35, 36, 34, 40, 37, 31, 33, 33, 39, 36, 32, 33, 34, 35, 47, 34, 37, 33, 35, 34, 32, 32, 36, 40, 42, 33, 30, 36, 39, 32, 41, 34, 42, 40, 41, 37, 45, 31, 41, 43, 33, 45, 37, 42, 47, 34, 34, 48, 36, 40, 33, 40, 39, 36, 43, 40, 40, 41, 42, 42, 36, 46, 52, 43, 50, 48, 44, 40, 39, 44, 48, 43, 45, 40, 55, 44, 39, 48, 38, 38, 45, 48, 46, 45, 45, 47, 51, 44, 47, 50, 41, 41, 44, 40, 46, 40, 40, 52, 43, 50, 43, 41, 40, 50, 40, 45, 40, 43, 45, 51, 46, 48, 50, 45, 47, 39, 47, 59, 43, 41, 50, 50, 48, 43, 49, 38, 47, 48, 51, 51, 46, 45, 47, 48, 55, 43, 43, 42, 37, 52, 52, 47, 41, 54, 52, 48, 54, 52, 43, 46, 49, 46, 43, 47, 47, 47, 48, 56, 49, 41, 57, 46, 38, 54, 44, 35, 35, 52, 35, 43, 50, 50, 45, 58, 54, 44, 42, 47, 47, 50, 49, 41, 45, 53, 45, 40, 53, 53, 45, 56, 40, 47, 49, 52, 48, 44, 41, 55, 50, 55, 52, 37, 42, 39, 51, 56, 45, 50, 47, 50, 53, 43, 52, 43, 46, 51, 55, 48, 55, 50, 43, 51, 47, 47, 52, 53, 65, 52, 58, 45, 55, 56, 55, 44, 46, 54, 54, 45, 56, 50, 42, 49, 47, 55, 43, 58, 47, 49, 58, 53, 46, 43, 60, 42, 51, 54, 50, 45, 45, 46, 53, 44, 43, 50, 43, 40, 41, 52, 55, 52, 52, 43, 37, 51, 47, 51, 49, 52, 43, 55, 48, 51, 51, 52, 56, 58, 46, 48, 43, 38, 48, 43, 50, 42, 50, 58, 46, 45, 47, 46, 55, 52, 42, 53, 51, 46, 53, 48, 57, 51, 54, 51, 57, 49, 48, 48, 47, 50, 40, 50, 47, 48, 49, 53, 51, 46, 45, 52, 52, 52, 47, 44, 45, 56, 47, 47, 51, 50, 49, 43]
reward_f8_ft    = [15, 15, 18, 21, 17, 16, 17, 16, 15, 19, 16, 23, 15, 25, 26, 25, 25, 24, 21, 25, 23, 18, 27, 25, 27, 23, 27, 30, 27, 29, 28, 29, 33, 24, 34, 30, 28, 33, 29, 29, 28, 43, 33, 28, 36, 29, 35, 37, 42, 35, 39, 42, 28, 33, 31, 44, 26, 27, 33, 33, 40, 33, 43, 41, 47, 42, 45, 36, 37, 31, 27, 39, 40, 40, 36, 40, 38, 32, 21, 43, 32, 32, 36, 39, 38, 37, 41, 46, 34, 42, 37, 36, 47, 33, 41, 34, 42, 38, 36, 41, 42, 46, 42, 37, 36, 41, 43, 42, 39, 32, 37, 38, 39, 41, 36, 43, 37, 45, 39, 36, 29, 31, 48, 29, 42, 42, 38, 45, 44, 37, 40, 34, 37, 34, 45, 34, 40, 42, 47, 41, 47, 35, 35, 38, 30, 36, 40, 40, 38, 39, 41, 41, 37, 39, 32, 38, 34, 33, 42, 37, 42, 51, 48, 36, 49, 35, 45, 42, 38, 39, 34, 38, 46, 28, 44, 43, 45, 34, 37, 40, 43, 41, 45, 38, 41, 42, 37, 38, 41, 36, 39, 51, 45, 43, 39, 53, 47, 40, 49, 41, 48, 39, 41, 39, 47, 50, 42, 36, 34, 39, 53, 48, 44, 43, 40, 37, 30, 52, 36, 39, 44, 33, 43, 42, 42, 42, 41, 48, 40, 39, 33, 47, 41, 42, 38, 35, 36, 37, 35, 41, 34, 38, 42, 33, 35, 38, 38, 35, 42, 37, 35, 39, 45, 34, 35, 45, 49, 45, 36, 44, 38, 31, 45, 43, 38, 29, 47, 43, 36, 46, 52, 46, 50, 37, 48, 40, 34, 40, 45, 47, 40, 40, 44, 48, 35, 43, 47, 40, 48, 50, 49, 41, 44, 48, 44, 39, 46, 39, 41, 44, 41, 42, 43, 49, 45, 48, 41, 45, 46, 48, 51, 38, 44, 43, 41, 45, 45, 37, 51, 46, 51, 40, 38, 44, 29, 42, 48, 38, 37, 40, 40, 44, 39, 48, 53, 53, 40, 45, 47, 47, 40, 45, 35, 43, 40, 44, 43, 49, 41, 37, 34, 42, 46, 34, 40, 36, 37, 34, 39, 49, 37, 43, 39, 39, 48, 55, 33, 45, 51, 42, 40, 39, 47, 50, 46, 40, 43, 45, 52, 43, 42, 41, 46, 47, 44, 45, 42, 46, 52, 35, 45, 47, 41, 37, 43, 52, 42, 45, 39, 39]

## ## ## ## ## ## ## SMOOTHED PLOTTING ## ## ## ## ## ## ##

x = np.arange(1, len(reward_f1) + 1)

# perform cubic spline interpolation
x_new = np.linspace(x.min(), x.max(), 50)  # 10 represents number of points to make it smoother
spl_f1 = make_interp_spline(x, reward_f1, k=3)
spl_f10 = make_interp_spline(x, reward_f10, k=3)
spl_f1_ft = make_interp_spline(x, reward_f1_ft, k=3)
spl_f5_ft = make_interp_spline(x, reward_f5_ft, k=3)
spl_f7_ft = make_interp_spline(x, reward_f7_ft, k=3)
spl_f8_ft = make_interp_spline(x, reward_f8_ft, k=3)

# generate smoothed data
reward_f1_smooth = spl_f1(x_new)
reward_f10_smooth = spl_f10(x_new)
reward_f1_ft_smooth = spl_f1_ft(x_new)
reward_f5_ft_smooth = spl_f5_ft(x_new)
reward_f7_ft_smooth = spl_f7_ft(x_new)
reward_f8_ft_smooth = spl_f8_ft(x_new)

# create the plot
plt.figure(figsize=(10, 6))

x_A = [x * (40000/50) for x in range(len(x_new))]
x_new = x_A

plt.subplot(1, 2, 1)
plt.plot(x_new, reward_f1_smooth, label='[1, 10, 100, 1]')
plt.plot(x_new, reward_f1_ft_smooth, label='[1, 10, 100, 1], first_touch')

plt.plot(x_new, reward_f10_smooth, label='[0, 10, 100, 0]')
plt.plot(x_new, reward_f5_ft_smooth, label='[1, 10, 1, 1], first touch')
plt.plot(x_new, reward_f7_ft_smooth, label='[10, 10, 100, 1], first touch')
plt.plot(x_new, reward_f8_ft_smooth, label='[100, 10, 100, 1], first touch')

print(len(x_new))

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


x_axis = [x * 100 for x in range(len(reward_f1))]

'''
# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, reward_f1, label='f1')
plt.plot(x_axis, reward_f1_ft, label='f1_ft')
plt.plot(x_axis, reward_f5_ft, label='f5_ft')
plt.plot(x_axis, reward_f7_ft, label='f7_ft')
plt.plot(x_axis, reward_f8_ft, label='f8_ft')
plt.plot(x_axis, reward_t, label='t')

'''

plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different reward functions')
plt.legend()
#plt.grid(True)

## ## ## ## ## ## ## POLYNOMIAL REGRESSION ## ## ## ## ## ##

# transform the input features into polynomial features
degree = 3
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(np.array(x_axis).reshape(-1, 1))

# initialize the regressor
regressor = LinearRegression()

# fit polynomial regression models
regressor.fit(x_poly, reward_f1)
poly_reward_f1 = regressor.predict(x_poly)

regressor.fit(x_poly, reward_f10)
poly_reward_f10 = regressor.predict(x_poly)

regressor.fit(x_poly, reward_f1_ft)
poly_reward_f1_ft = regressor.predict(x_poly)

regressor.fit(x_poly, reward_f5_ft)
poly_reward_f5_ft = regressor.predict(x_poly)

regressor.fit(x_poly, reward_f7_ft)
poly_reward_f7_ft = regressor.predict(x_poly)

regressor.fit(x_poly, reward_f8_ft)
poly_reward_f8_ft = regressor.predict(x_poly)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_reward_f1, label='[1, 10, 100, 1]')
plt.plot(x_axis, poly_reward_f1_ft, label='[1, 10, 100, 1], first touch')

plt.plot(x_axis, poly_reward_f10, label='[0, 10, 100, 0]')
plt.plot(x_axis, poly_reward_f5_ft, label='[1, 10, 1, 1], first touch')
plt.plot(x_axis, poly_reward_f7_ft, label='[10, 10, 100, 1], first touch')
plt.plot(x_axis, poly_reward_f8_ft, label='[100, 10, 100, 1], first touch')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()