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

reward_f1   = 
reward_ft   = 
reward_t    = 

x_axis = [x * 100 for x in range(len(reward_f1))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, reward_f1, label='factor = [1, 10, 100, 1]')
plt.plot(x_axis, reward_ft, label='factor = [1, 10, 100 (first touch), 1]')
plt.plot(x_axis, reward_t, label='10*winner + 50*closeness_puck - (1-touch_puck) + (touch_puck*first_touch*step) + 100*puck_direction')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different reward functions')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f1)
poly_reward_f1 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_ft)
poly_reward_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), poly_reward_ft)
poly_reward_t = regressor.predict(np.array(x_axis).reshape(-1, 1))


# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_reward_f1, label='factor = [1, 10, 100, 1]')
plt.plot(x_axis, poly_reward_ft, label='factor = [1, 10, 100 (first touch), 1]')
plt.plot(x_axis, poly_reward_t, label='10*winner + 50*closeness_puck - (1-touch_puck) + (touch_puck*first_touch*step) + 100*puck_direction')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()