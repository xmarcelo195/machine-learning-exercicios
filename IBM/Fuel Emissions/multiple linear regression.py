import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

pd.set_option('display.max_columns', None) # Exibir todas as Colunas

df = pd.read_csv('FuelConsumptionCo2.csv')
print(df.columns)

data = df[['ENGINESIZE',
           'CYLINDERS',
           'FUELCONSUMPTION_CITY',
           'FUELCONSUMPTION_HWY',
           'FUELCONSUMPTION_COMB',
           'CO2EMISSIONS']]
'''
#plotar grafico para visualizar tendencias
for coluna in data.columns:
    plt.scatter(df['{}'.format(coluna)], df.CO2EMISSIONS, color='blue')
    plt.ylabel('Emissão de Co2')
    plt.xlabel('{}'.format(coluna))
    plt.show()
'''

# Divisão dados para treino
msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]

# Sklearn
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE',
                               'CYLINDERS',
                               'FUELCONSUMPTION_CITY',
                               'FUELCONSUMPTION_HWY']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Coefficientes
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# test
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
x_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

# Soma do erro residual
print("Erro: %2f" % np.mean((y_hat - y_test)**2))

#variancia
print('Variancia: %2f' % regr.score(x_test, y_test))
