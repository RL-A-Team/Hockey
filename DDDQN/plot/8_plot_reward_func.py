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

reward_r    = [3, 2, 4, 1, 6, 4, 5, 6, 4, 5, 6, 6, 5, 12, 6, 6, 9, 8, 18, 12, 12, 16, 17, 14, 16, 14, 9, 10, 19, 15, 19, 17, 20, 19, 16, 18, 21, 23, 12, 20]
reward_f1   = [2, 5, 9, 22, 17, 14, 18, 13, 14, 17, 20, 19, 21, 19, 19, 20, 11, 16, 29, 18, 26, 19, 24, 26, 23, 28, 24, 22, 23, 23, 22, 31, 28, 24, 19, 30, 24, 28, 28, 29]
reward_f2   = [3, 6, 1, 13, 3, 6, 5, 6, 8, 12, 6, 13, 12, 19, 13, 21, 19, 20, 19, 15, 11, 20, 13, 18, 19, 13, 18, 23, 16, 13, 12, 21, 13, 18, 13, 19, 20, 18, 18, 13]
reward_f3   = [4, 4, 9, 2, 10, 12, 6, 12, 15, 14, 19, 16, 16, 15, 12, 12, 16, 19, 21, 25, 24, 22, 16, 16, 7, 24, 17, 13, 24, 17, 21, 21, 20, 18, 17, 10, 24, 24, 15, 26]
reward_f1_ft = [2, 1, 16, 13, 13, 15, 24, 18, 13, 18, 17, 14, 20, 13, 20, 17, 22, 19, 20, 16, 11, 18, 15, 22, 24, 21, 17, 30, 20, 27, 26, 20, 22, 30, 21, 17, 24, 26, 26, 31]
reward_t    = [15, 15, 15, 8, 16, 14, 11, 16, 16, 18, 12, 24, 21, 24, 17, 26, 15, 19, 25, 27, 13, 21, 26, 19, 22, 17, 15, 23, 25, 32, 19, 36, 23, 27, 31, 31, 30, 32, 30, 29]

x_axis = [x * 100 for x in range(len(reward_r))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, reward_r, label='reward')
plt.plot(x_axis, reward_f1, label='factor = [1, 10, 100, 1]')
plt.plot(x_axis, reward_f2, label='factor = [10, 1, 1, 10]')
plt.plot(x_axis, reward_f3, label='factor = [10, 5, 1, 1]')
plt.plot(x_axis, reward_f1_ft, label='factor = [1, 10, 100 (first touch), 1]')
plt.plot(x_axis, reward_t, label='10*winner + 50*closeness_puck - (1-touch_puck) + (touch_puck*first_touch*step) + 100*puck_direction')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different reward functions')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_r)
poly_reward_r = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f1)
poly_reward_f1 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f2)
poly_reward_f2 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f3)
poly_reward_f3 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f1_ft)
poly_reward_f1_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_t)
poly_reward_t = regressor.predict(np.array(x_axis).reshape(-1, 1))


# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_reward_r, label='reward')
plt.plot(x_axis, poly_reward_f1, label='factor = [1, 10, 100, 1]')
plt.plot(x_axis, poly_reward_f2, label='factor = [10, 1, 1, 10]')
plt.plot(x_axis, poly_reward_f3, label='factor = [10, 5, 1, 1]')
plt.plot(x_axis, poly_reward_f1_ft, label='factor = [1, 10, 100 (firt touch), 1]')
plt.plot(x_axis, poly_reward_t, label='10*winner + 50*closeness_puck - (1-touch_puck) + (touch_puck*first_touch*step) + 100*puck_direction')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()