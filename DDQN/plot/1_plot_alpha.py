import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
episodes = 4000 mode = NORMAL
hidden_dim = [300,300]

tau = 5e-3         
learning_rate = 1e-3 
discount = 0.99      
batch_size = 256     
epsilon = 1e-6          
max_size = 100000
reward = reward
'''

alpha_0_1 = [12, 4, 4, 5, 12, 11, 13, 15, 9, 15, 8, 18, 23, 12, 13, 19, 11, 20, 17, 16, 13, 16, 17, 22, 16, 17, 26, 17, 13, 14, 14, 19, 18, 24, 24, 19, 23, 16, 26, 13] # 11.69 min
alpha_0_2 = [7, 3, 4, 2, 5, 6, 5, 14, 8, 4, 13, 10, 15, 12, 12, 18, 12, 15, 19, 20, 15, 15, 10, 13, 16, 17, 16, 14, 13, 18, 21, 15, 25, 14, 17, 16, 13, 18, 20, 12] # 11.83 min
alpha_0_3 = [5, 8, 18, 16, 15, 12, 11, 12, 15, 8, 16, 11, 20, 7, 20, 16, 10, 10, 10, 18, 11, 6, 22, 8, 14, 13, 10, 20, 13, 21, 15, 16, 19, 17, 14, 22, 14, 13, 14, 16] # 12.54 min
alpha_0_4 = [6, 3, 4, 3, 4, 5, 12, 4, 3, 5, 7, 10, 4, 13, 6, 6, 13, 8, 7, 5, 13, 5, 7, 10, 15, 11, 9, 9, 11, 9, 20, 12, 14, 18, 17, 12, 12, 17, 14, 13] # 15.77 min, local

x_axis = [x * 100 for x in range(len(alpha_0_1))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, alpha_0_1, label='0.1')
plt.plot(x_axis, alpha_0_2, label='0.2')
plt.plot(x_axis, alpha_0_3, label='0.3')
plt.plot(x_axis, alpha_0_4, label='0.4')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different alpha values')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), alpha_0_1)
poly_alpha_0_1 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), alpha_0_2)
poly_alpha_0_2 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), alpha_0_3)
poly_alpha_0_3 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), alpha_0_4)
poly_alpha_0_4 = regressor.predict(np.array(x_axis).reshape(-1, 1))

# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_alpha_0_1, label='0.1')
plt.plot(x_axis, poly_alpha_0_2, label='0.2')
plt.plot(x_axis, poly_alpha_0_3, label='0.3')
plt.plot(x_axis, poly_alpha_0_4, label='0.4')


# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()