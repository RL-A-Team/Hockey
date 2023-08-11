import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy.interpolate import make_interp_spline


'''
episodes = 10000 mode = NORMAL
hidden_dim = [300,300]
alpha = 0.1
tau = 1e-3
discount = 0.96
batch_size = 256
epsilon = 1e-4 
max_size = 1000000
'''

f1    = [16, 38, 38, 30, 30, 30, 31, 29, 35, 34, 27, 33, 33, 32, 35, 25, 31, 29, 31, 38]  
f1_ft = [33, 41, 33, 32, 45, 32, 37, 41, 31, 31, 36, 36, 40, 35, 39, 24, 35, 34, 30, 38]   

## ## ## ## ## ## ## SMOOTHED PLOTTING ## ## ## ## ## ## ##

x = np.arange(1, len(f1) + 1)

# perform cubic spline interpolation
x_new = np.linspace(x.min(), x.max(), 10) # 10 represents number of points to make it smoother
spl_f1 = make_interp_spline(x, f1, k=3)
spl_f1_ft = make_interp_spline(x, f1_ft, k=3)

# generate smoothed data
f1_smooth = spl_f1(x_new)
f1_ft_smooth = spl_f1_ft(x_new)

# create the plot
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(x_new, f1_smooth, label='f1', linewidth=2)
plt.plot(x_new, f1_ft_smooth, label='f1 ft', linewidth=2)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


x_axis = [x * 100 for x in range(len(f1))]

plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different opponents, different training volumes')
plt.legend()
plt.grid(True)

## ## ## ## ## ## ## POLYNOMIAL REGRESSION ## ## ## ## ## ##

# transform the input features into polynomial features
degree = 3
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(np.array(x_axis).reshape(-1, 1))

# initialize the regressor
regressor = LinearRegression()

# fit polynomial regression models
regressor.fit(x_poly, f1)
poly_f1 = regressor.predict(x_poly)

regressor.fit(x_poly, f1_ft)
poly_f1_ft = regressor.predict(x_poly)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_f1, label='f1')
plt.plot(x_axis, poly_f1_ft, label='f1 ft')


# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()