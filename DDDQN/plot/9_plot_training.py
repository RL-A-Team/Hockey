import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
episodes = 4000 mode = NORMAL
hidden_dim = [300,300]
alpha = 0.1
tau = 1e-3
discount = 0.96
batch_size = 256
epsilon = 1e-4 
max_size = 1000000
'''

reward_f1   = [3, 6, 13, 15, 13, 14, 17, 15, 15, 19, 12, 23, 26, 13, 20, 17, 17, 16, 15, 19, 25, 23, 14, 27, 26, 20, 16, 17, 18, 18, 21, 25, 23, 25, 24, 14, 37, 24, 23, 25, 16, 24, 38, 21, 19, 33, 28, 27, 31, 30, 36, 31, 36, 31, 27, 35, 29, 33, 31, 29, 34, 32, 30, 28, 40, 35, 32, 41, 29, 32, 32, 42, 35, 29, 32, 30, 40, 34, 41, 40, 41, 42, 37, 33, 40, 36, 37, 37, 39, 41, 38, 36, 34, 38, 36, 42, 38, 31, 33, 40, 34, 33, 40, 32, 35, 36, 33, 36, 27, 41, 41, 35, 37, 34, 35, 42, 36, 35, 35, 37, 33, 46, 34, 32, 25, 42, 40, 39, 42, 37, 40, 41, 44, 40, 32, 36, 49, 44, 45, 42, 44, 38, 47, 52, 54, 42, 40, 44, 39, 36, 44, 37, 47, 54, 47, 50, 41, 38, 54, 55, 48, 44, 44, 51, 42, 40, 44, 43, 40, 49, 38, 43, 45, 36, 45, 40, 41, 44, 40, 45, 41, 54, 51, 38, 42, 48, 45, 54, 59, 46, 41, 48, 36, 53, 49, 48, 39, 48, 47, 46, 43, 41, 47, 39, 52, 40, 51, 45, 45, 38, 43, 59, 47, 48, 49, 38, 46, 49, 45, 31, 45, 48, 50, 43, 50, 44, 33, 49, 42, 47, 41, 47, 42, 38, 43, 40, 39, 44, 46, 42, 41, 41, 43, 44, 34, 33, 39, 40, 41, 35, 38, 44, 47, 41, 47, 37, 38, 49, 41, 45, 47, 48, 45, 44, 51, 48, 48, 50, 48, 51, 51, 39, 45, 53, 46, 51, 50, 57, 44, 42, 50, 46, 57, 56, 61, 44, 51, 45, 52, 58, 48, 52, 49, 57, 45, 48, 52, 44, 51, 46, 50, 53, 63, 53, 55, 52, 59, 53, 51, 55, 49, 47, 49, 53, 50, 53, 55, 50, 49, 56, 54, 49, 57, 55, 49, 45, 52, 43, 44, 48, 46, 48, 44, 42, 52, 47, 46, 47, 58, 48, 55, 50, 53, 53, 41, 53, 53, 52, 51, 42, 51, 54, 48, 50, 35, 54, 50, 55, 46, 58, 53, 51, 65, 52, 60, 62, 59, 58, 61, 56, 57, 61, 58, 50, 58, 51, 61, 53, 58, 56, 61, 63, 59, 55, 62, 67, 56, 57, 57, 60, 50, 67, 57, 46, 57, 57, 51, 61, 60, 64, 67, 60, 62, 63, 58, 71, 72, 62, 67, 58, 67, 72, 69, 68, 61, 66, 68, 65, 58, 55, 58, 61, 67, 65, 68, 75, 57, 68, 60, 55, 65, 58, 66, 69, 70, 64, 72, 64, 60, 63, 73, 60, 56, 71, 70, 69, 71, 64, 67, 75, 64, 75, 71, 76, 79, 74, 75, 69, 70, 80, 63, 78, 71, 58, 69, 72, 65, 75, 65, 75, 68, 63, 62, 61, 76, 69, 62, 63, 62, 60, 56, 67, 48, 53, 53, 53, 42, 61, 51, 55, 51, 63, 47, 49, 47, 47, 44, 55, 53, 44, 31, 46, 45, 43, 53, 58, 47, 46, 48, 48, 58, 59, 37, 43, 46, 42, 32, 37, 36, 40, 49, 42, 41, 40, 42, 38, 34, 38, 41, 44, 46, 42, 36, 47, 41, 39, 33, 40, 53, 48, 52, 51, 49, 44, 42, 65, 37, 47, 47, 55, 43, 54, 49, 52, 48, 55, 56, 53, 48, 40, 44, 41, 51, 53, 54, 22, 37, 41, 52, 46, 59, 56, 44, 43, 51, 60, 61, 55, 44, 48, 53, 41, 34, 54, 53, 64, 50, 60, 62, 47, 44, 60, 54, 46, 55, 47, 42, 53, 53, 46, 39, 37, 56, 46, 54, 52, 53, 64, 31, 45, 39, 44, 36, 38, 47, 41, 35, 26, 38, 14, 32, 30, 34, 31, 17, 31, 36, 39, 38, 27, 18, 37, 43, 50, 34, 17, 27, 39, 33, 41, 38, 39, 45, 45, 57, 45, 29, 38, 40, 37, 52, 57, 55, 50, 39, 46, 34, 36, 46, 44, 44, 27, 38, 34, 29, 48, 50, 43, 44, 44, 46, 43, 37, 27, 34, 41, 31, 38, 39, 38, 49, 34, 33, 37, 47, 27, 37, 32, 42, 57, 44, 26, 64, 55, 31, 36, 30, 24, 32, 47, 28, 38, 26, 41, 18, 46, 26, 18, 35, 23, 14, 23, 36, 23, 10, 11, 16, 13, 27, 27, 14, 25, 25, 28, 32, 24, 29, 24, 26, 20, 18, 21, 25, 22, 40, 27, 34, 34, 30, 35, 31, 41, 39, 28, 56, 31, 37, 37, 63, 42, 42, 34, 38, 29, 39, 27, 20, 34, 50, 21, 38, 23, 32, 28, 25, 44, 38, 38, 38, 41, 30, 47, 33, 47, 21, 14, 20, 22, 20, 24, 9, 16, 16, 27, 26, 28, 32, 21, 40, 35, 22, 10, 18, 16, 13, 18, 25, 17, 29, 23, 33, 11, 18, 27, 24, 30, 20, 17, 12, 21, 24, 15, 25, 16, 23, 27, 25, 20, 26, 27, 28, 39, 41, 35, 44, 36, 38, 37, 39, 15, 43, 34, 47, 28, 46, 47, 51, 31, 58, 58, 41, 50, 32, 41, 39, 33, 24, 36, 33, 40, 31, 27, 29, 41, 57, 35, 32, 26, 31, 33, 36, 40, 18, 18, 19, 31, 18, 30, 34, 31, 29, 22, 20, 21, 23, 23, 13, 14, 30, 13, 16, 26, 20, 15, 16, 15, 21, 21, 21, 13, 11, 23, 18, 13, 24, 28, 35, 25, 19, 31, 27, 21, 30, 33, 40, 52, 37, 39, 23, 34, 25, 39, 33, 36, 53, 48, 49, 30, 33, 32, 28, 19, 32, 32, 24, 22, 14, 30, 28, 23, 34, 40, 23, 31, 23, 33, 23, 21, 15, 20, 21, 20, 19, 11, 33, 28, 28, 21, 22, 21, 25, 19, 22, 20, 33, 21, 32, 22, 32, 26, 20, 18, 27, 22, 31, 30, 27, 27, 18, 24, 20, 28, 11, 19, 19, 17, 26, 34, 28, 17, 22, 17, 13, 11, 19, 20, 18, 10, 16, 23, 10, 10, 33, 19, 45, 43, 27, 18, 22, 31]
reward_t    = [7, 10, 15, 17, 17, 14, 15, 17, 16, 13, 14, 25, 13, 20, 21, 19, 25, 24, 25, 20, 31, 24, 35, 19, 28, 23, 31, 27, 26, 21, 36, 22, 26, 28, 29, 35, 31, 24, 26, 34, 33, 39, 32, 24, 38, 29, 31, 36, 28, 30, 26, 24, 22, 24, 28, 18, 28, 29, 31, 32, 26, 25, 30, 26, 25, 24, 30, 23, 41, 33, 33, 29, 33, 34, 27, 30, 32, 29, 35, 28, 24, 27, 33, 32, 31, 31, 30, 36, 38, 31, 32, 29, 44, 34, 35, 27, 32, 38, 33, 37, 29, 33, 37, 39, 30, 39, 46, 37, 31, 32, 33, 32, 35, 35, 40, 33, 50, 30, 39, 38, 34, 30, 36, 36, 45, 40, 34, 33, 30, 33, 45, 33, 35, 41, 33, 41, 43, 36, 41, 35, 28, 38, 29, 41, 40, 34, 38, 50, 41, 44, 42, 31, 35, 37, 37, 44, 39, 36, 36, 31, 40, 35, 42, 42, 37, 44, 36, 42, 48, 28, 48, 28, 34, 31, 38, 39, 33, 27, 44, 38, 39, 34, 38, 32, 40, 37, 42, 37, 38, 35, 43, 40, 36, 38, 43, 32, 41, 39, 37, 39, 38, 38, 45, 40, 38, 36, 37, 38, 45, 36, 38, 50, 53, 32, 39, 46, 42, 45, 39, 36, 44, 36, 49, 41, 42, 41, 42, 44, 29, 44, 39, 40, 44, 49, 51, 43, 40, 32, 42, 47, 39, 39, 48, 42, 51, 46, 52, 45, 50, 45, 43, 47, 49, 41, 41, 29, 44, 47, 47, 36, 43, 46, 46, 43, 38, 39, 41, 38, 38, 54, 56, 35, 39, 44, 43, 48, 50, 53, 42, 38, 52, 59, 51, 40, 43, 54, 45, 41, 52, 44, 53, 45, 44, 51, 42, 43, 45, 51, 45, 46, 55, 41, 48, 46, 54, 57, 44, 47, 45, 44, 45, 49, 43, 47, 44, 32, 42, 43, 43, 50, 44, 36, 42, 49, 42, 43, 51, 48, 47, 44, 55, 46, 40, 45, 53, 53, 46, 57, 44, 52, 54, 41, 41, 36, 47, 48, 45, 51, 42, 38, 49, 53, 43, 46, 40, 45, 41, 50, 47, 46, 46, 43, 53, 43, 42, 50, 39, 52, 37, 33, 50, 47, 45, 39, 47, 44, 50, 48, 44, 50, 43, 48, 48, 37, 45, 50, 41, 44, 44, 45, 46, 44, 55, 44, 39, 48, 39, 51, 52, 42, 38, 45, 38, 41, 51, 51, 47, 42, 42, 51, 49, 54, 43, 46, 54, 41, 43, 45, 45, 46, 39, 47, 46, 52, 47, 45, 49, 45, 56, 43, 55, 48, 42, 42, 51, 54, 49, 55, 48, 54, 41, 42, 51, 52, 52, 47, 48, 46, 43, 52, 47, 45, 38, 45, 46, 50, 49, 50, 37, 48, 43, 45, 44, 43, 45, 48, 48, 39, 51, 49, 47, 49, 39, 49, 50, 39, 53, 49, 42, 45, 47, 45, 42, 49, 39, 35, 52, 52, 42, 48, 49, 50, 42, 45, 42, 45, 53, 46, 46, 50, 44, 42, 53, 40, 41, 36, 53, 52, 54, 46, 47, 48, 47, 49, 38, 61, 50, 54, 46, 51, 40, 44, 49, 40, 52, 42, 46, 50, 57, 42, 49, 41, 45, 52, 46, 59, 56, 50, 46, 48, 47, 54, 37, 43, 52, 45, 39, 51, 54, 50, 42, 50, 46, 41, 53, 46, 49, 53, 44, 56, 42, 41, 52, 50, 51, 47, 47, 47, 50, 44, 46, 49, 50, 44, 41, 48, 43, 52, 48, 52, 49, 51, 43, 46, 47, 48, 46, 45, 46, 49, 38, 44, 46, 44, 35, 41, 40, 40, 43, 53, 53, 47, 52, 50, 47, 52, 51, 48, 46, 49, 47, 55, 48, 54, 43, 46, 51, 53, 47, 50, 51, 49, 53, 45, 50, 40, 39, 59, 48, 46, 51, 40, 37, 50, 55, 49, 50, 49, 50, 42, 52, 56, 51, 46, 57, 51, 59, 53, 44, 46, 52, 49, 46, 35, 53, 51, 43, 56, 44, 50, 47, 44, 48, 45, 47, 50, 48, 43, 50, 44, 47, 45, 51, 49, 46, 44, 49, 50, 54, 46, 54, 48, 41, 52, 38, 58, 56, 42, 44, 49, 46, 43, 42, 54, 45, 50, 49, 62, 48, 45, 49, 46, 40, 53, 57, 43, 38, 49, 45, 56, 41, 45, 38, 48, 58, 43, 51, 50, 47, 50, 43, 45, 54, 51, 51, 54, 44, 43, 46, 46, 44, 55, 47, 48, 47, 56, 46, 56, 43, 49, 46, 54, 57, 46, 39, 50, 50, 53, 40, 40, 57, 49, 48, 46, 34, 41, 49, 56, 41, 46, 45, 48, 48, 48, 48, 43, 47, 46, 49, 54, 52, 49, 41, 47, 52, 48, 54, 49, 53, 47, 53, 48, 54, 47, 49, 46, 47, 47, 44, 51, 42, 43, 46, 54, 53, 46, 42, 40, 49, 50, 40, 51, 48, 48, 49, 47, 57, 52, 49, 49, 54, 47, 53, 55, 52, 46, 54, 51, 55, 47, 48, 54, 53, 39, 42, 48, 62, 45, 49, 47, 46, 53, 45, 55, 39, 39, 55, 61, 45, 50, 56, 50, 50, 46, 51, 59, 49, 60, 57, 53, 52, 51, 47, 51, 51, 48, 43, 38, 37, 48, 50, 52, 50, 63, 56, 46, 52, 59, 40, 43, 51, 43, 48, 51, 52, 47, 51, 51, 56, 54, 49, 44, 53, 38, 47, 48, 45, 48, 53, 44, 38, 51, 50, 49, 43, 43, 44, 46, 52, 44, 50, 46, 48, 46, 44, 54, 49, 51, 48, 46, 42, 45, 52, 47, 51, 50, 45, 40, 46, 45, 39, 42, 38, 46, 47, 46, 49, 37, 42, 36, 48, 46, 42, 40, 41, 50, 56, 38, 45, 43, 45, 53, 56, 51, 46, 47, 48, 55, 53, 48, 47, 48, 42, 42, 45, 39, 54, 52, 53, 45, 49, 38, 43, 51, 52, 41, 54, 57, 41, 52, 35, 58, 51, 56, 49, 58, 38, 50, 46, 51, 51, 56, 45, 49, 51, 41, 41, 51, 50, 48, 57, 60, 51, 51, 53, 50, 43, 43, 55, 45]


x_axis = [x * 100 for x in range(len(reward_f1))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, reward_f1, label='factor = [1, 10, 100, 1]')
plt.plot(x_axis, reward_t, label='10*winner + 50*closeness_puck - (1-touch_puck) + (touch_puck*first_touch*step) + 100*puck_direction')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different reward functions')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f1)
poly_reward_f1 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_t)
poly_reward_t = regressor.predict(np.array(x_axis).reshape(-1, 1))


# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_reward_f1, label='factor = [1, 10, 100, 1]')
plt.plot(x_axis, poly_reward_t, label='10*winner + 50*closeness_puck - (1-touch_puck) + (touch_puck*first_touch*step) + 100*puck_direction')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()