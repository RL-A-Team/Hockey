import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
episodes = 4000 mode = NORMAL

alpha = 0.2          
tau = 5e-3         
learning_rate = 1e-3 
discount = 0.99      
batch_size = 256     
epsilon = 1e-6          
max_size = 100000
'''

hd_100_100 = [10, 14, 13, 12, 14, 11, 7, 14, 11, 9, 9, 15, 9, 10, 19, 19, 11, 14, 17, 14, 13, 14, 15, 17, 21, 22, 16, 24, 12, 13, 19, 14, 14, 22, 16, 19, 11, 20, 16, 15]   # 11.66 min x
hd_200_200 = [5, 5, 7, 10, 6, 9, 8, 11, 5, 13, 13, 7, 9, 11, 7, 10, 16, 9, 12, 20, 12, 20, 16, 19, 17, 12, 12, 16, 15, 27, 18, 15, 18, 19, 17, 17, 18, 19, 23, 19]          # 14.01 min x
hd_300_300 = [4, 10, 6, 7, 9, 7, 7, 3, 5, 7, 6, 13, 13, 7, 18, 15, 16, 16, 10, 14, 20, 17, 18, 12, 17, 17, 12, 19, 12, 17, 17, 18, 23, 25, 12, 21, 20, 20, 18, 19]          # 11.92 min :)
hd_300_200 = [6, 7, 7, 10, 9, 11, 13, 9, 9, 9, 10, 15, 11, 11, 19, 15, 16, 8, 17, 15, 18, 18, 17, 22, 17, 16, 15, 19, 16, 22, 16, 25, 19, 24, 19, 15, 15, 17, 13, 16]       # 12.23 min x
hd_200_100 = [10, 9, 12, 7, 5, 8, 9, 7, 10, 13, 13, 8, 21, 17, 14, 6, 11, 16, 16, 16, 14, 12, 16, 11, 20, 12, 21, 18, 14, 23, 20, 17, 16, 19, 19, 22, 16, 14, 26, 21]       # 12.16 min x
hd_100_50 =  [7, 4, 8, 9, 3, 12, 9, 9, 6, 9, 11, 11, 12, 8, 11, 12, 14, 10, 14, 12, 11, 17, 17, 9, 14, 9, 10, 20, 11, 18, 18, 12, 19, 22, 14, 16, 17, 22, 12, 19]           # 11.33 min x
hd_300_200_100 = [7, 9, 14, 6, 9, 16, 7, 8, 10, 12, 6, 9, 17, 8, 16, 9, 9, 15, 14, 19, 15, 17, 12, 21, 16, 15, 12, 20, 19, 20, 20, 17, 18, 16, 20, 13, 15, 21, 19, 15]      # 11.46 min x

x_axis = [x * 100 for x in range(len(hd_100_100))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, hd_100_100, label='[100,100]')
plt.plot(x_axis, hd_200_200, label='[200,200]')
plt.plot(x_axis, hd_300_300, label='[300,300]')
plt.plot(x_axis, hd_300_200, label='[300,200]')
plt.plot(x_axis, hd_200_100, label='[200,100]')
plt.plot(x_axis, hd_100_50, label='[100,50]')
plt.plot(x_axis, hd_300_200_100, label='[300,200,100]')

plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different hidden_dim values')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), hd_100_100)
poly_hd_100_100 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), hd_200_200)
poly_hd_200_200 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), hd_300_300)
poly_hd_300_300 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), hd_300_200)
poly_hd_300_200 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), hd_200_100)
poly_hd_200_100 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), hd_100_50)
poly_hd_100_50 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), hd_300_200_100)
poly_hd_300_200_100 = regressor.predict(np.array(x_axis).reshape(-1, 1))

# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_hd_100_100, label='[100,100]')
plt.plot(x_axis, poly_hd_200_200, label='[200,200]')
plt.plot(x_axis, poly_hd_300_300, label='[300,300]')
plt.plot(x_axis, poly_hd_300_200, label='[300,200]')
plt.plot(x_axis, poly_hd_200_100, label='[200,100]')
plt.plot(x_axis, poly_hd_100_50, label='[100,50]')
plt.plot(x_axis, poly_hd_300_200_100, label='[300,200,100]')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()