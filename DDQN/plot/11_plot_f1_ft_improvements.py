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
baseline      = [38, 46, 40, 35, 27, 38, 44, 33, 37, 31]
weak_5000     = [31, 38, 43, 36, 40, 36, 27, 40, 36, 38]
strong_5000   = [34, 31, 37, 34, 39, 37, 35, 34, 34, 34]
weak_10000    = [37, 36, 39, 38, 35, 39, 35, 42, 37, 34]
strong_10000  = [32, 39, 42, 35, 34, 32, 36, 40, 43, 34]

## ## ## ## ## ## ## SMOOTHED PLOTTING ## ## ## ## ## ## ##

x = np.arange(1, len(baseline) + 1)

# perform cubic spline interpolation
x_new = np.linspace(x.min(), x.max(), 5) # 10 represents number of points to make it smoother
bs = make_interp_spline(x, baseline, k=3)
spl_w5 = make_interp_spline(x, weak_5000, k=3)
spl_s5 = make_interp_spline(x, strong_5000, k=3)
spl_w10 = make_interp_spline(x, weak_10000, k=3)
spl_s10 = make_interp_spline(x, strong_10000, k=3)

# generate smoothed data
baseline_smooth = bs(x_new)
weak_5000_smooth = spl_w5(x_new)
strong_5000_smooth = spl_s5(x_new)
weak_10000_smooth = spl_w10(x_new)
strong_10000_smooth = spl_s10(x_new)

# create the plot
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(x_new, baseline_smooth, label='baseline', linewidth=2)
plt.plot(x_new, weak_5000_smooth, label='weak 5000', linewidth=2)
plt.plot(x_new, strong_5000_smooth, label='strong 5000', linewidth=2)
plt.plot(x_new, weak_10000_smooth, label='weak 10000', linewidth=2)
plt.plot(x_new, strong_10000_smooth, label='strong 10000', linewidth=2)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


x_axis = [x * 100 for x in range(len(baseline))]

plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('different opponents, different training volumes')
plt.legend()
plt.grid(True)

## ## ## ## ## ## ## POLYNOMIAL REGRESSION ## ## ## ## ## ##

# transform the input features into polynomial features
degree = 2
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(np.array(x_axis).reshape(-1, 1))

# initialize the regressor
regressor = LinearRegression()

# fit polynomial regression models
regressor.fit(x_poly, baseline)
poly_baseline = regressor.predict(x_poly)

regressor.fit(x_poly, weak_5000)
poly_weak_5000 = regressor.predict(x_poly)

regressor.fit(x_poly, strong_5000)
poly_strong_5000 = regressor.predict(x_poly)

regressor.fit(x_poly, weak_10000)
poly_weak_10000 = regressor.predict(x_poly)

regressor.fit(x_poly, strong_10000)
poly_strong_10000 = regressor.predict(x_poly)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

# Plot the regression lines
plt.subplot(1, 2, 2)
plt.plot(x_axis, poly_baseline, label='baseline')
plt.plot(x_axis, poly_weak_5000, label='weak 5000')
plt.plot(x_axis, poly_strong_5000, label='strong 5000')
plt.plot(x_axis, poly_weak_10000, label='weak 10000')
plt.plot(x_axis, poly_strong_10000, label='strong 10000')


# Add labels and title
plt.xlabel('episodes')
plt.ylabel('win percentage')
plt.title('regression lines')

plt.legend()

plt.show()