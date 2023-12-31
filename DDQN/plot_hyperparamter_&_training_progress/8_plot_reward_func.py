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
max_size = 1000000
'''

reward_r    = [3, 2, 4, 1, 6, 4, 5, 6, 4, 5, 6, 6, 5, 12, 6, 6, 9, 8, 18, 12, 12, 16, 17, 14, 16, 14, 9, 10, 19, 15, 19, 17, 20, 19, 16, 18, 21, 23, 12, 20]
reward_f1   = [2, 5, 9, 22, 17, 14, 18, 13, 14, 17, 20, 19, 21, 19, 19, 20, 11, 16, 29, 18, 26, 19, 24, 26, 23, 28, 24, 22, 23, 23, 22, 31, 28, 24, 19, 30, 24, 28, 28, 29]
reward_f2   = [3, 6, 1, 13, 3, 6, 5, 6, 8, 12, 6, 13, 12, 19, 13, 21, 19, 20, 19, 15, 11, 20, 13, 18, 19, 13, 18, 23, 16, 13, 12, 21, 13, 18, 13, 19, 20, 18, 18, 13]
reward_f3   = [4, 4, 9, 2, 10, 12, 6, 12, 15, 14, 19, 16, 16, 15, 12, 12, 16, 19, 21, 25, 24, 22, 16, 16, 7, 24, 17, 13, 24, 17, 21, 21, 20, 18, 17, 10, 24, 24, 15, 26]
reward_f4   = [12, 10, 9, 12, 11, 15, 18, 13, 10, 9, 15, 16, 17, 20, 12, 10, 12, 21, 17, 19, 17, 19, 19, 12, 23, 14, 24, 13, 25, 21, 20, 21, 24, 23, 27, 20, 22, 24, 18, 16]
reward_f5   = [9, 10, 16, 18, 20, 18, 19, 17, 14, 16, 21, 14, 18, 22, 21, 20, 15, 26, 24, 25, 21, 25, 17, 21, 35, 25, 26, 33, 23, 27, 30, 30, 20, 28, 28, 33, 28, 35, 27, 29]
reward_f6   = [16, 14, 15, 18, 12, 16, 20, 14, 17, 25, 21, 19, 18, 23, 16, 18, 17, 13, 13, 16, 20, 24, 16, 18, 18, 15, 29, 21, 22, 27, 28, 25, 24, 19, 17, 23, 23, 27, 21, 22]
reward_f7   = [9, 9, 8, 14, 15, 13, 8, 17, 11, 20, 12, 14, 14, 24, 19, 17, 22, 18, 16, 18, 19, 14, 15, 22, 19, 25, 22, 19, 27, 23, 23, 20, 22, 29, 29, 27, 29, 37, 28, 35]
reward_f8   = [7, 12, 10, 11, 10, 13, 11, 9, 17, 15, 18, 16, 16, 16, 14, 17, 22, 18, 20, 26, 27, 23, 25, 22, 26, 25, 24, 22, 27, 25, 22, 28, 27, 30, 25, 27, 31, 32, 30, 22]
reward_f9   = [4, 10, 12, 18, 9, 12, 14, 17, 15, 15, 17, 14, 13, 15, 24, 17, 15, 22, 20, 23, 20, 23, 22, 25, 14, 18, 17, 28, 24, 25, 21, 29, 27, 24, 20, 18, 21, 25, 23, 29]

reward_f10     = [11, 16, 23, 25, 16, 14, 12, 17, 19, 20, 13, 13, 24, 24, 16, 22, 20, 23, 23, 18, 20, 20, 28, 23, 25, 30, 30, 31, 31, 34, 28, 27, 31, 28, 35, 37, 30, 35, 34, 35]
reward_f11     = [3, 7, 12, 9, 10, 13, 11, 16, 15, 20, 25, 15, 21, 17, 14, 20, 24, 27, 21, 22, 22, 26, 17, 19, 29, 26, 32, 29, 33, 17, 29, 28, 36, 27, 29, 32, 40, 27, 30, 21]
reward_f10_ft  = [8, 7, 14, 15, 19, 9, 13, 19, 14, 15, 17, 20, 21, 13, 17, 24, 16, 23, 18, 30, 27, 23, 24, 22, 28, 21, 34, 21, 23, 29, 20, 27, 31, 30, 22, 27, 33, 27, 21, 26]
reward_f11_ft  = [7, 7, 16, 9, 11, 12, 13, 13, 17, 16, 18, 26, 18, 15, 20, 17, 14, 20, 10, 25, 20, 24, 18, 19, 22, 18, 18, 28, 16, 19, 20, 27, 23, 18, 20, 19, 25, 22, 27, 21]

reward_f1_ft = [2, 1, 16, 13, 13, 15, 24, 18, 13, 18, 17, 14, 20, 13, 20, 17, 22, 19, 20, 16, 11, 18, 15, 22, 24, 21, 17, 30, 20, 27, 26, 20, 22, 30, 21, 17, 24, 26, 26, 31]
reward_f5_ft = [15, 8, 11, 11, 12, 20, 18, 17, 21, 20, 21, 21, 22, 23, 12, 20, 26, 22, 22, 20, 27, 19, 25, 28, 26, 26, 27, 19, 25, 23, 28, 23, 34, 21, 25, 25, 24, 34, 35, 35]
reward_f7_ft = [10, 17, 19, 21, 17, 10, 23, 17, 22, 18, 19, 22, 18, 21, 22, 17, 21, 23, 22, 18, 24, 26, 17, 29, 27, 27, 35, 29, 22, 28, 42, 35, 28, 36, 28, 28, 31, 36, 34, 34]
reward_f8_ft = [6, 16, 16, 16, 25, 16, 19, 15, 21, 21, 19, 15, 13, 24, 19, 21, 21, 23, 26, 23, 24, 23, 26, 31, 26, 17, 28, 17, 23, 27, 23, 24, 27, 27, 36, 29, 28, 28, 34, 31]

reward_t    = [15, 15, 15, 8, 16, 14, 11, 16, 16, 18, 12, 24, 21, 24, 17, 26, 15, 19, 25, 27, 13, 21, 26, 19, 22, 17, 15, 23, 25, 32, 19, 36, 23, 27, 31, 31, 30, 32, 30, 29]

x_axis = [x * 100 for x in range(len(reward_r))]

# Plot the original data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, reward_r, label='reward')
plt.plot(x_axis, reward_f1, label='[1, 10, 100, 1]')
plt.plot(x_axis, reward_f1_ft, label='[1, 10, 100, 1], first_touch')

#plt.plot(x_axis, reward_f2, label='f2')
#plt.plot(x_axis, reward_f5, label='f5')

#plt.plot(x_axis, reward_f6, label='f6')
#plt.plot(x_axis, reward_f7, label='f7')

plt.plot(x_axis, reward_f10, label='[0, 10, 100, 0]')

plt.plot(x_axis, reward_f5_ft, label='[1, 10, 1, 1], first touch')
plt.plot(x_axis, reward_f7_ft, label='[10, 10, 100, 1], first touch')
plt.plot(x_axis, reward_f8_ft, label='[100, 10, 100, 1], first touch')


#plt.plot(x_axis, reward_f3, label='f3')
#plt.plot(x_axis, reward_f4, label='f4')
#plt.plot(x_axis, reward_f8, label='f8')
#plt.plot(x_axis, reward_f9, label='f9')

#plt.plot(x_axis, reward_f11, label='f11')



#plt.plot(x_axis, reward_f10_ft, label='f10_ft')
#plt.plot(x_axis, reward_f11_ft, label='f11_ft')


#plt.plot(x_axis, reward_t, label='t')


plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different reward functions')
plt.legend()

# Perform Linear Regression
regressor = LinearRegression()

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_r)
poly_reward_r = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f1)
poly_reward_f1 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f2)
poly_reward_f2 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f3)
poly_reward_f3 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f4)
poly_reward_f4 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f5)
poly_reward_f5 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f6)
poly_reward_f6 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f7)
poly_reward_f7 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f8)
poly_reward_f8 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f9)
poly_reward_f9 = regressor.predict(np.array(x_axis).reshape(-1, 1))


regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f10)
poly_reward_f10 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f10_ft)
poly_reward_f10_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f11)
poly_reward_f11 = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f11_ft)
poly_reward_f11_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))


regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f1_ft)
poly_reward_f1_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f5_ft)
poly_reward_f5_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f7_ft)
poly_reward_f7_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_f8_ft)
poly_reward_f8_ft = regressor.predict(np.array(x_axis).reshape(-1, 1))

regressor.fit(np.array(x_axis).reshape(-1, 1), reward_t)
poly_reward_t = regressor.predict(np.array(x_axis).reshape(-1, 1))


# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_reward_r, label='reward')
plt.plot(x_axis, poly_reward_f1, label='[1, 10, 100, 1]')
plt.plot(x_axis, poly_reward_f1_ft, label='[1, 10, 100, 1], first_touch')

plt.plot(x_axis, poly_reward_f10, label='[0, 10, 100, 0]')

plt.plot(x_axis, poly_reward_f5_ft, label='[1, 10, 1, 1], first touch')
plt.plot(x_axis, poly_reward_f7_ft, label='[10, 10, 100, 1], first touch')
plt.plot(x_axis, poly_reward_f8_ft, label='[100, 10, 100, 1], first touch')

#plt.plot(x_axis, poly_reward_f2, label='f2')
#plt.plot(x_axis, poly_reward_f3, label='f3')
#plt.plot(x_axis, poly_reward_f4, label='f4')
#plt.plot(x_axis, poly_reward_f5, label='f5')
#plt.plot(x_axis, poly_reward_f6, label='f6')
#plt.plot(x_axis, poly_reward_f7, label='f7')
#plt.plot(x_axis, poly_reward_f8, label='f8')
#plt.plot(x_axis, poly_reward_f9, label='f9')


#plt.plot(x_axis, poly_reward_f10_ft, label='f10_ft')
#plt.plot(x_axis, poly_reward_f11, label='f11')
#plt.plot(x_axis, poly_reward_f11_ft, label='f11_ft')


#plt.plot(x_axis, poly_reward_t, label='t')

# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()