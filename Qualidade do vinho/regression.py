import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics

df = pd.read_csv('winequality_clean.csv')
# print(df.columns)

'''
# Visualizar histograma
viz = df
viz.hist()
plt.tight_layout()
plt.show()
'''

'''
# Visualizar Dados
data = df.columns.drop(['QUALITY'])
for coluna in data:

    plt.scatter(df['{}'.format(coluna)], df.QUALITY, color='blue')
    plt.ylabel('Qualidade')
    plt.xlabel('{}'.format(coluna))

'''

# Variáveis com maior correlação
inputs = ['VOLATILE_ACIDITY',
          'SULPHATES',
          'ALCOHOL']

# separando dataframe em test e treino
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

#sklearn
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[inputs])

train_y = np.asanyarray(train[['QUALITY']])
regr.fit(train_x, train_y)

# coeficientes
print('Coeficiente = ', regr.coef_)
print('Intercept = ', regr.intercept_)

'''
# plot output
plt.scatter(df.PH, df.QUALITY, color='blue')
plt.plot(train_x, train_x * regr.coef_ + regr.intercept_, '-r')
plt.xlabel('PH')
plt.ylabel('QUALIDADE')
plt.show()
'''

#test
y_ = regr.predict(test[inputs])
y = np.asanyarray(test[['QUALITY']])
x = np.asanyarray(test[inputs])

# Soma do erro residual
print("Erro: %2f" % np.mean((y_ - y)**2))

#variancia
print('Variancia: %2f' % regr.score(x, y))

print('R2', metrics.r2_score(y, y_))
print('root mean', metrics.mean_squared_error(y, y_))
