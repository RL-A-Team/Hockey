import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
episodes = 4000 mode = NORMAL
hidden_dim = [300,300]
alpha = 0.1
tau = 1e-3
       
batch_size = 256     
epsilon = 1e-6          
max_size = 100000
reward = reward
'''

discount_0_96 = [2, 1, 1, 7, 6, 0, 2, 4, 6, 1, 5, 11, 11, 6, 9, 14, 14, 14, 15, 14, 11, 13, 16, 15, 15, 15, 11, 13, 12, 17, 19, 21, 17, 16, 14, 18, 23, 22, 25, 23]
discount_0_97 = [6, 3, 8, 5, 2, 5, 2, 9, 4, 6, 11, 8, 18, 12, 9, 12, 10, 18, 17, 12, 20, 21, 13, 18, 17, 20, 21, 22, 23, 23, 22, 19, 11, 21, 22, 21, 10, 20, 16, 17]
discount_0_98 = [11, 6, 3, 0, 4, 2, 6, 4, 7, 8, 6, 8, 11, 14, 17, 15, 18, 12, 16, 16, 18, 12, 12, 17, 12, 15, 17, 16, 15, 21, 17, 14, 21, 19, 26, 17, 14, 25, 18, 17]
discount_0_99 = [11, 6, 3, 0, 4, 2, 6, 4, 7, 8, 6, 8, 11, 14, 17, 15, 18, 12, 16, 16, 18, 12, 12, 17, 12, 15, 17, 16, 15, 21, 17, 14, 21, 19, 26, 17, 14, 25, 18, 17]
discount_0_999 = [2, 2, 1, 2, 4, 0, 2, 4, 4, 5, 4, 2, 5, 2, 6, 9, 11, 8, 14, 13, 14, 8, 8, 14, 10, 5, 7, 8, 13, 9, 12, 10, 15, 6, 11, 12, 22, 12, 9, 18]

x_axis = [x * 100 for x in range(len(discount_0_96))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, discount_0_96, label='0.96')
plt.plot(x_axis, discount_0_97, label='0.97')
plt.plot(x_axis, discount_0_98, label='0.98')
plt.plot(x_axis, discount_0_99, label='0.99')
plt.plot(x_axis, discount_0_999, label='0.999')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different dicount values')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), discount_0_96)
poly_discount_0_96 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), discount_0_97)
poly_discount_0_97 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), discount_0_98)
poly_discount_0_98 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), discount_0_99)
poly_discount_0_99 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), discount_0_999)
poly_discount_0_999 = regressor.predict(np.array(x_axis).reshape(-1, 1))

# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_discount_0_96, label='0.96')
plt.plot(x_axis, poly_discount_0_97, label='0.97')
plt.plot(x_axis, poly_discount_0_98, label='0.98')
plt.plot(x_axis, poly_discount_0_99, label='0.99')
plt.plot(x_axis, poly_discount_0_999, label='0.999')



# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()