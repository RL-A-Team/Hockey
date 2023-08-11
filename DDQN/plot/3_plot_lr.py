import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
episodes = 4000 mode = NORMAL
hidden_dim = [300,300]
alpha = 0.1
tau = 1e-3
       
discount = 0.99      
batch_size = 256     
epsilon = 1e-6          
max_size = 100000
reward = reward
'''

lr_1e_3 = [2, 3, 4, 6, 4, 6, 8, 2, 7, 7, 10, 7, 12, 17, 18, 21, 13, 13, 14, 21, 17, 17, 13, 14, 9, 12, 18, 25, 21, 20, 29, 16, 13, 25, 20, 23, 25, 18, 17, 22]
lr_2e_3 = [4, 1, 2, 0, 4, 5, 4, 1, 4, 2, 4, 5, 5, 4, 3, 10, 11, 18, 21, 14, 9, 7, 14, 10, 11, 8, 11, 4, 4, 2, 5, 5, 4, 10, 7, 6, 8, 3, 7, 8]
lr_3e_3 =  [8, 6, 1, 5, 2, 5, 3, 6, 5, 10, 13, 15, 9, 5, 22, 8, 20, 12, 17, 14, 14, 17, 13, 15, 13, 17, 18, 22, 21, 13, 13, 13, 21, 16, 16, 11, 22, 16, 14, 18]

lr_1e_2 = [3, 6, 3, 5, 4, 3, 4, 6, 3, 5, 3, 4, 4, 3, 3, 5, 6, 6, 7, 3, 3, 4, 1, 2, 4, 4, 3, 5, 6, 4, 2, 4, 3, 3, 1, 6, 3, 6, 4, 3]
lr_1e_4 = [2, 3, 2, 3, 5, 6, 5, 5, 4, 8, 4, 12, 6, 6, 5, 9, 8, 15, 9, 9, 11, 15, 15, 14, 11, 14, 12, 11, 15, 11, 10, 11, 15, 14, 11, 19, 15, 17, 22, 17]
lr_1e_5 =  [13, 14, 6, 7, 11, 8, 5, 10, 5, 10, 3, 9, 6, 8, 12, 2, 12, 11, 6, 21, 5, 16, 12, 10, 12, 12, 12, 14, 7, 16, 9, 10, 13, 11, 10, 12, 16, 12, 8, 6]

x_axis = [x * 100 for x in range(len(lr_1e_3))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, lr_1e_3, label='1e-3')
plt.plot(x_axis, lr_2e_3, label='2e-3')
plt.plot(x_axis, lr_3e_3, label='3e-3')
plt.plot(x_axis, lr_1e_2, label='1e-2')
plt.plot(x_axis, lr_1e_4, label='1e-4')
plt.plot(x_axis, lr_1e_5, label='1e-5')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different learning rate values')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), lr_1e_3)
poly_lr_1e_3 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), lr_2e_3)
poly_lr_2e_3 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), lr_3e_3)
poly_lr_3e_3 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), lr_1e_2)
poly_lr_1e_2 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), lr_1e_4)
poly_lr_1e_4 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), lr_1e_5)
poly_lr_1e_5 = regressor.predict(np.array(x_axis).reshape(-1, 1))

# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_lr_1e_3, label='1e-3')
plt.plot(x_axis, poly_lr_2e_3, label='2e-3')
plt.plot(x_axis, poly_lr_3e_3, label='3e-3')
plt.plot(x_axis, poly_lr_1e_2, label='1e-2')
plt.plot(x_axis, poly_lr_1e_4, label='1e-4')
plt.plot(x_axis, poly_lr_1e_5, label='1e-5')


# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()