import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
episodes = 4000 mode = NORMAL
hidden_dim = [300,300]
alpha = 0.1
tau = 1e-3
discount = 0.96

epsilon = 1e-6          
max_size = 100000
reward = reward
'''

batch_size_128 = [6, 5, 16, 13, 9, 10, 13, 15, 11, 8, 11, 14, 15, 7, 12, 15, 12, 9, 15, 17, 18, 18, 12, 13, 13, 13, 12, 19, 22, 17, 11, 15, 22, 22, 18, 25, 22, 18, 14, 19]
batch_size_256 = [4, 7, 9, 11, 15, 10, 15, 12, 11, 12, 7, 13, 12, 16, 11, 14, 25, 20, 7, 16, 16, 17, 19, 19, 23, 13, 23, 22, 16, 16, 21, 17, 21, 26, 21, 21, 18, 20, 20, 21]
batch_size_512 = [10, 7, 12, 12, 9, 12, 9, 10, 12, 12, 11, 14, 9, 9, 12, 13, 16, 17, 19, 17, 19, 19, 23, 16, 18, 15, 20, 18, 21, 24, 20, 15, 24, 19, 15, 17, 11, 14, 23, 23]

x_axis = [x * 100 for x in range(len(batch_size_128))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, batch_size_128, label='128')
plt.plot(x_axis, batch_size_256, label='256')
plt.plot(x_axis, batch_size_512, label='512')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different batch size values')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), batch_size_128)
poly_batch_size_128 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), batch_size_256)
poly_batch_size_256 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), batch_size_512)
poly_batch_size_512 = regressor.predict(np.array(x_axis).reshape(-1, 1))


# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_batch_size_128, label='128')
plt.plot(x_axis, poly_batch_size_256, label='256')
plt.plot(x_axis, poly_batch_size_512, label='512')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()