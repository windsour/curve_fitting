import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def model_function(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.array([10, 26, 44, 70, 90])
ydata = np.array([4.2, 3.8, 3.5, 3.2, 3.0])
initial_guess = [1.0, 0.1, 0.5]

params, covariance = curve_fit(model_function, xdata, ydata, p0=initial_guess)

print("Optimal parameters:", params)
print("Covariance of parameters:\n", covariance)

# 绘图相关代码...
x_fit = np.linspace(min(xdata), max(xdata), 100)
y_fit = model_function(x_fit, *params)

plt.scatter(xdata, ydata, label='Data', color='red')
plt.plot(x_fit, y_fit, label='Fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params))
plt.xlabel('X data')
plt.ylabel('Y data')
plt.title('Curve Fitting Example')
plt.legend()
plt.show()

# 计算R²
y_pred = model_function(xdata, *params)
ss_res = np.sum((ydata - y_pred) ** 2)
ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print("R-squared:", r_squared)