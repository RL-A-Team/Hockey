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

max_size_1000       = 
max_size_10000      = 
max_size_100000     = 
max_size_1000000    = 
max_size_10000000   = 

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