import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('Iris.csv')
# print(df.head(10))


# Transformar valores
# tot['Sex'] = tot['Sex'].map({'male': 0, 'female': 1})

df['Species'] = df['Species'].map({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})
# print(df.Species)

# 'Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'

#separar treino e test
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
Y = df[['Species']].values.ravel()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print('Train set: ', x_train.shape, y_train.shape)
print('Test set: ', x_test.shape, y_test.shape)

#treino
k = 6
neigh = KNeighborsClassifier(n_neighbors=k).fit(X, Y)
y_ = neigh.predict(x_test)

#accuracy

print(accuracy_score(y_test, y_))
