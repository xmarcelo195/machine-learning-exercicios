import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


#importar csv
df = pd.read_csv('CleanTrain.csv')

#limpar nan
df = df.fillna(-1)

#criar arrays X e Y
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values
y = df[['Survived']].values.ravel()

#normalizar valores de x
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

'''
#dividir df test e treino
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print('Train set: ', x_train.shape, y_train.shape)
print('Test set: ', x_test.shape, y_test.shape)
'''

#treino
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X, y)
# y_ = neigh.predict(x_tes) <- uso apenas se tiver dividido o df em teste e treino

#accuracy
print('Train set Accuracy: ', metrics.accuracy_score(y, neigh.predict(X)))

'''
#test new k
Ks = 50
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = []
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(x_train, y_train)
    yhat = neigh.predict(x_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(y_ == y_test) / np.sqrt(yhat.shape[0])

#plot

plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)'''

df_test = pd.read_csv('test_clin.csv')
# replace string to number
# df_test.replace(to_replace=['set', 'test'], value=[1, 2])

X_test = df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))

df_id = df_test[['PassengerId']] #df with ids

Survived = []
for i in neigh.predict(X_test):
    Survived.append(i)


print(len(df_id))
print(len(X_test))
df_id = df_id.assign(Survived=Survived)

df_id.to_csv('Submission.csv', index=False)
