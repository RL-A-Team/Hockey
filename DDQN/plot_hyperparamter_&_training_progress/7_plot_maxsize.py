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

reward = reward
'''

max_size_1000       = [14, 19, 14, 17, 14, 6, 20, 16, 18, 11, 15, 17, 16, 12, 19, 12, 13, 15, 16, 13, 17, 14, 9, 16, 14, 21, 21, 14, 24, 31, 19, 20, 23, 21, 19, 16, 16, 21, 19, 19]
max_size_10000      = [8, 12, 17, 11, 10, 12, 9, 10, 18, 17, 13, 14, 16, 6, 13, 16, 25, 20, 17, 17, 14, 19, 19, 18, 9, 15, 18, 16, 17, 20, 16, 17, 10, 18, 18, 24, 20, 21, 16, 22]
max_size_100000     = [15, 8, 16, 8, 12, 7, 6, 12, 16, 11, 12, 11, 19, 16, 16, 21, 15, 19, 10, 17, 17, 17, 19, 15, 24, 21, 20, 16, 14, 19, 26, 24, 23, 14, 20, 12, 20, 11, 20, 19]
max_size_1000000    = [7, 5, 8, 2, 7, 11, 7, 11, 10, 6, 6, 16, 11, 22, 11, 18, 15, 12, 11, 24, 14, 14, 18, 16, 16, 19, 17, 15, 21, 20, 13, 22, 20, 19, 21, 22, 16, 24, 17, 20]
max_size_10000000   = [8, 13, 15, 14, 14, 15, 12, 20, 19, 18, 11, 13, 12, 20, 21, 18, 16, 17, 21, 17, 18, 25, 26, 23, 14, 23, 19, 17, 16, 13, 16, 17, 22, 22, 14, 14, 20, 20, 19, 25]

x_axis = [x * 100 for x in range(len(max_size_1000))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, max_size_1000, label='1000')
plt.plot(x_axis, max_size_10000, label='10000')
plt.plot(x_axis, max_size_100000, label='100000')
plt.plot(x_axis, max_size_1000000, label='1000000')
plt.plot(x_axis, max_size_10000000, label='10000000')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different max_size values')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), max_size_1000)
poly_max_size_1000 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), max_size_10000)
poly_max_size_10000 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), max_size_100000)
poly_max_size_100000 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), max_size_1000000)
poly_max_size_1000000 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), max_size_10000000)
poly_max_size_10000000 = regressor.predict(np.array(x_axis).reshape(-1, 1))


# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_max_size_1000, label='1000')
plt.plot(x_axis, poly_max_size_10000, label='10000')
plt.plot(x_axis, poly_max_size_100000, label='100000')
plt.plot(x_axis, poly_max_size_1000000, label='1000000')
plt.plot(x_axis, poly_max_size_10000000, label='10000000')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()