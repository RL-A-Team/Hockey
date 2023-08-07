import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 1 - 0.99 * np.exp(-6 * np.exp(-0.3 * x))

x = np.linspace(-1, 50, 100)
y = f(x)
print(y)
plt.plot(x, y)
plt.grid()
plt.show()