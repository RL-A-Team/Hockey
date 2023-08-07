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

eps_1e_4 = 
eps_1e_5 = 
eps_1e_6 = 
eps_1e_7 = 
eps_1e_8 = 

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