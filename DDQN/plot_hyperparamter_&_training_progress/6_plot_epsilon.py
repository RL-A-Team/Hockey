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

max_size = 100000
reward = reward
'''

eps_1e_4 = [6, 4, 5, 5, 7, 8, 10, 7, 14, 10, 15, 17, 16, 13, 17, 16, 22, 15, 17, 17, 18, 19, 11, 22, 15, 19, 17, 16, 23, 16, 22, 22, 19, 22, 21, 22, 12, 14, 19, 17]
eps_1e_5 = [3, 5, 7, 3, 8, 5, 7, 6, 4, 7, 11, 7, 8, 11, 15, 12, 11, 13, 8, 13, 13, 16, 13, 16, 22, 19, 15, 10, 18, 18, 15, 25, 17, 19, 18, 16, 20, 15, 18, 20]
eps_1e_6 = [15, 10, 13, 9, 20, 11, 14, 18, 16, 12, 13, 14, 18, 13, 15, 19, 13, 14, 17, 16, 17, 8, 19, 20, 17, 14, 16, 13, 19, 16, 21, 23, 20, 14, 22, 19, 24, 17, 18, 25]
eps_1e_7 = [1, 1, 3, 2, 2, 3, 2, 3, 4, 4, 8, 9, 10, 15, 8, 17, 11, 17, 16, 16, 13, 6, 15, 13, 12, 19, 17, 25, 15, 8, 12, 18, 14, 10, 23, 22, 17, 19, 19, 22]
eps_1e_8 = [3, 1, 3, 6, 3, 7, 4, 3, 4, 8, 3, 4, 10, 4, 6, 15, 13, 10, 12, 12, 11, 13, 10, 19, 14, 12, 18, 18, 13, 18, 13, 18, 15, 10, 13, 17, 20, 17, 15, 22]


x_axis = [x * 100 for x in range(len(eps_1e_4))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, eps_1e_4, label='1e-4')
plt.plot(x_axis, eps_1e_5, label='1e-5')
plt.plot(x_axis, eps_1e_6, label='1e-6')
plt.plot(x_axis, eps_1e_7, label='1e-7')
plt.plot(x_axis, eps_1e_8, label='1e-8')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different epsilon values')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), eps_1e_4)
poly_eps_1e_4 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), eps_1e_5)
poly_eps_1e_5 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), eps_1e_6)
poly_eps_1e_6 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), eps_1e_7)
poly_eps_1e_7 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), eps_1e_8)
poly_eps_1e_8 = regressor.predict(np.array(x_axis).reshape(-1, 1))


# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_eps_1e_4, label='1e-4')
plt.plot(x_axis, poly_eps_1e_5, label='1e-5')
plt.plot(x_axis, poly_eps_1e_6, label='1e-6')
plt.plot(x_axis, poly_eps_1e_7, label='1e-7')
plt.plot(x_axis, poly_eps_1e_8, label='1e-8')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()