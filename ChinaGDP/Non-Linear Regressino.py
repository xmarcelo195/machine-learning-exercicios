import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def func(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


df = pd.read_csv('china_gdp.csv')
x_data, y_data = (df['Year'].values, df['Value'].values)

'''
# print dos dados
plt.scatter(x_data, y_data, color='blue')
plt.show()
'''

xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

'''

'''
# divisao dos dados
msk = np.random.rand(len(df)) < 0.8
x_train = xdata[msk]
y_train = ydata[msk]
x_test = xdata[~msk]
y_test = ydata[~msk]

#treino fit
popt, pcov = curve_fit(func, x_train, y_train)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

y_ = func(x_test, *popt)

# Dados d erro
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_))
print("Residual sum of squares (MSE): %.5f" % mean_absolute_error(y_test, y_))
print("R2-score: %.2f" % r2_score(y_test, y_))


x = np.linspace(1960, 2015, 55)
x = x/max(x)
y = func(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()