import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
episodes = 4000 mode = NORMAL
hidden_dim = [300,300]
alpha = 0.1

       
learning_rate = 1e-3 
discount = 0.99      
batch_size = 256     
epsilon = 1e-6          
max_size = 100000
reward = reward
'''

tau_5e_3 =  [9, 7, 10, 5, 12, 10, 12, 14, 12, 11, 12, 10, 20, 14, 17, 25, 14, 10, 16, 14, 18, 18, 13, 15, 16, 15, 27, 27, 17, 28, 20, 20, 14, 19, 15, 14, 24, 19, 23, 18] # 11.11 min, local

tau_3e_3 =  [5, 3, 11, 6, 14, 9, 11, 10, 6, 5, 13, 9, 6, 7, 13, 10, 12, 9, 14, 7, 12, 11, 17, 20, 13, 14, 11, 14, 12, 21, 12, 18, 17, 18, 15, 18, 23, 18, 13, 12] # 12.35 min
tau_7e_3 =  [11, 12, 10, 11, 10, 6, 12, 9, 13, 15, 12, 13, 17, 11, 9, 12, 12, 12, 11, 7, 9, 9, 7, 15, 8, 9, 16, 9, 13, 14, 12, 11, 13, 6, 12, 25, 17, 20, 17, 6] # 12.64 min

tau_5e_2 = [14, 11, 12, 7, 13, 16, 13, 16, 14, 21, 22, 15, 18, 17, 22, 10, 15, 20, 19, 21, 18, 22, 22, 27, 16, 20, 18, 13, 15, 15, 19, 18, 13, 24, 18, 18, 17, 21, 15, 16] # 12.24 min
tau_5e_4 = [9, 10, 13, 13, 6, 13, 12, 9, 11, 6, 9, 10, 9, 16, 13, 14, 11, 16, 17, 16, 14, 15, 13, 15, 15, 18, 14, 16, 14, 12, 14, 13, 22, 12, 21, 14, 12, 14, 17, 19] # 18.90 min, local

x_axis = [x * 100 for x in range(len(tau_5e_3))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, tau_5e_3, label='5e-3')
plt.plot(x_axis, tau_3e_3, label='3e-3')
plt.plot(x_axis, tau_7e_3, label='7e-3')
plt.plot(x_axis, tau_5e_2, label='5e-2')
plt.plot(x_axis, tau_5e_4, label='5e-4')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different tau values')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), tau_5e_3)
poly_tau_5e_3 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), tau_3e_3)
poly_tau_3e_3 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), tau_7e_3)
poly_tau_7e_3 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), tau_5e_2)
poly_tau_5e_2 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), tau_5e_4)
poly_tau_5e_4 = regressor.predict(np.array(x_axis).reshape(-1, 1))

# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_tau_5e_3, label='5e-3')
plt.plot(x_axis, poly_tau_3e_3, label='3e-3')
plt.plot(x_axis, poly_tau_7e_3, label='7e-3')
plt.plot(x_axis, poly_tau_5e_2, label='5e-2')
plt.plot(x_axis, poly_tau_5e_4, label='5e-4')


# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()