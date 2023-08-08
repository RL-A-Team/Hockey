import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
reward_f1_ft    = [4, 10, 16, 18, 14, 20, 16, 23, 25, 16, 30, 25, 26, 24, 20, 16, 22, 14, 27, 26, 29, 25, 18, 19, 21, 22, 30, 30, 32, 33, 26, 28, 36, 32, 35, 31, 39, 33, 41, 35, 37, 33, 28, 36, 39, 30, 31, 38, 47, 31, 31, 34, 38, 33, 27, 31, 38, 38, 38, 40, 38, 37, 39, 31, 30, 37, 27, 34, 37, 34, 39, 34, 29, 41, 39, 32, 33, 34, 37, 36, 36, 47, 47, 34, 32, 49, 46, 37, 40, 24, 38, 37, 42, 47, 44, 39, 44, 46, 44, 40, 40, 42, 41, 49, 46, 44, 39, 51, 40, 56, 50, 44, 37, 47, 42, 41, 34, 37, 42, 44, 37, 39, 49, 48, 45, 46, 50, 41, 50, 36, 39, 38, 39, 45, 43, 46, 43, 50, 38, 44, 44, 45, 44, 49, 50, 50, 45, 54, 40, 43, 43, 43, 62, 47, 55, 44, 54, 49, 47, 51, 44, 52, 58, 52, 49, 38, 49, 50, 50, 48, 45, 44, 44, 50, 49, 37, 46, 43, 56, 53, 48, 51, 37, 45, 55, 52, 54, 54, 51, 45, 46, 45, 60, 57, 45, 47, 56, 47, 44, 47, 51, 49, 44, 49, 45, 54, 45, 39, 51, 40, 35, 47, 49, 42, 53, 48, 44, 51, 43, 44, 56, 45, 42, 38, 45, 50, 45, 49, 47, 46, 48, 68, 37, 60, 57, 59, 55, 40, 52, 43, 51, 51, 53, 43, 57, 48, 49, 46, 52, 46, 55, 51, 50, 45, 47, 51, 49, 52, 48, 54, 46, 51, 51, 45, 49, 48, 48, 52, 57, 50, 58, 55, 51, 57, 51, 47, 50, 51, 43, 45, 40, 50, 47, 47, 51, 54, 49, 48, 46, 46, 58, 58, 40, 49, 50, 53, 52, 43, 44, 49, 49, 53, 47, 43, 42, 47, 53, 50, 50, 46, 56, 48, 53, 53, 48, 47, 45, 44, 38, 48, 42, 53, 38, 47, 48, 33, 40, 54, 46, 48, 50, 44, 52, 47, 43, 45, 46, 54, 47, 50, 54, 47, 54, 51, 46, 54, 48, 40, 48, 47, 50, 51, 50, 52, 44, 49, 52, 46, 55, 42, 53, 38, 45, 50, 55, 41, 45, 40, 50, 47, 48, 50, 50, 52, 43, 45, 50, 44, 50, 49, 47, 52, 46, 45, 52, 51, 52, 57, 50, 46, 46, 51, 48, 57, 50, 50, 41, 47, 49, 57]
reward_f5_ft    =
reward_f7_ft    =
reward_f8_ft    =
reward_t        = [5, 8, 17, 17, 16, 10, 14, 16, 12, 13, 19, 23, 21, 21, 11, 16, 17, 18, 17, 13, 20, 18, 31, 19, 22, 24, 23, 29, 21, 28, 34, 21, 29, 22, 28, 30, 31, 25, 29, 32, 23, 22, 31, 31, 30, 25, 34, 29, 42, 22, 27, 27, 32, 29, 31, 29, 24, 30, 32, 31, 29, 25, 28, 27, 34, 35, 23, 28, 26, 30, 26, 32, 28, 30, 25, 35, 33, 29, 25, 30, 27, 33, 25, 35, 34, 31, 27, 40, 36, 35, 36, 39, 42, 34, 39, 40, 33, 31, 27, 40, 44, 36, 39, 32, 28, 36, 38, 41, 31, 31, 36, 44, 25, 30, 41, 41, 35, 40, 29, 39, 32, 33, 43, 34, 35, 32, 27, 26, 26, 37, 35, 36, 30, 41, 36, 32, 29, 38, 36, 35, 32, 34, 32, 38, 33, 30, 37, 44, 35, 35, 37, 40, 45, 39, 36, 45, 37, 49, 34, 38, 37, 43, 36, 48, 36, 47, 36, 41, 38, 41, 37, 43, 35, 41, 37, 41, 47, 37, 47, 37, 40, 28, 45, 39, 44, 35, 45, 38, 40, 44, 39, 40, 36, 48, 46, 43, 44, 33, 41, 32, 48, 41, 43, 51, 43, 46, 45, 39, 39, 36, 38, 41, 35, 49, 46, 50, 48, 42, 39, 44, 51, 40, 42, 41, 47, 45, 45, 27, 49, 42, 37, 42, 43, 41, 33, 41, 47, 43, 45, 41, 37, 39, 36, 43, 41, 37, 52, 42, 38, 48, 42, 44, 46, 35, 44, 39, 37, 43, 48, 38, 47, 38, 46, 44, 39, 39, 46, 41, 48, 40, 45, 45, 45, 44, 44, 52, 33, 40, 39, 46, 32, 47, 42, 42, 36, 43, 48, 55, 42, 50, 42, 45, 39, 34, 40, 39, 48, 44, 32, 45, 47, 37, 44, 49, 50, 39, 51, 46, 40, 54, 48, 43, 56, 44, 48, 42, 43, 40, 53, 52, 48, 40, 41, 31, 43, 38, 42, 47, 38, 42, 40, 45, 46, 43, 52, 54, 47, 54, 40, 45, 50, 57, 50, 44, 48, 42, 36, 52, 54, 44, 47, 43, 39, 47, 43, 43, 49, 48, 48, 55, 52, 41, 53, 37, 44, 50, 46, 35, 38, 35, 47, 55, 42, 39, 44, 43, 47, 51, 41, 64, 60, 43, 37, 45, 57, 43, 51, 46, 48, 48, 53, 50, 44, 59, 46, 45, 52, 54, 41, 54]


x_axis = [x * 100 for x in range(len(reward_f1))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, reward_f1, label='f1')
plt.plot(x_axis, reward_f1_ft, label='f1_ft')
plt.plot(x_axis, reward_f5_ft, label='f5_ft')
plt.plot(x_axis, reward_f7_ft, label='f7_ft')
plt.plot(x_axis, reward_f8_ft, label='f8_ft')
plt.plot(x_axis, reward_t, label='t')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different reward functions')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f1)
poly_reward_f1 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f1_ft)
poly_reward_f1_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f5_ft)
poly_reward_f5_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f7_ft)
poly_reward_f7_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f8_ft)
poly_reward_f8_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_t)
poly_reward_t = regressor.predict(np.array(x_axis).reshape(-1, 1))


# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_reward_f1, label='f1')
plt.plot(x_axis, poly_reward_f1_ft, label='f1_ft')
plt.plot(x_axis, poly_reward_f5_ft, label='f5_ft')
plt.plot(x_axis, poly_reward_f7_ft, label='f7_ft')
plt.plot(x_axis, poly_reward_f8_ft, label='f8_ft')
plt.plot(x_axis, poly_reward_t, label='t')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()