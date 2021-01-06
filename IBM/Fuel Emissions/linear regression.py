import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# importando dataset
df = pd.read_csv('/home/marcelo/PycharmProjects/Data/Coursera/IBM/Fuel Emissions/FuelConsumptionCo2.csv')
df_clean = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

'''
Mostrar o Histrograma
viz = df_clean
viz.hist()
plt.tight_layout()
plt.show()

plotar grafico engine x co2
plt.scatter(df_clean.ENGINESIZE, df_clean.CO2EMISSIONS, color='blue')
plt.xlabel('tamanho do motor')
plt.ylabel('Emissão de CO2')
plt.show()

plotar grafico cilindro x co2
plt.scatter(df_clean.CYLINDERS, df_clean.CO2EMISSIONS, color='green', marker=None)
plt.xlabel('Cilindros')
plt.ylabel('Emissão de CO2')
plt.show()

'''

# separando dataframe em test e treino
msk = np.random.rand(len(df_clean)) < 0.8
train = df_clean[msk]
test = df_clean[~msk]

# sklearn
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# coeficientes
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# plot output
plt.scatter(df_clean.ENGINESIZE, df_clean.CO2EMISSIONS, color='blue')
plt.plot(train_x, train_x * regr.coef_ + regr.intercept_, '-r')
plt.xlabel('Tamanho do Motor')
plt.ylabel('Emissão CO2')
plt.show()

# validação
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Erro Média Absoluta: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))
