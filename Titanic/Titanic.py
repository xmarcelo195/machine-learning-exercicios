import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

# configuração pd
pd.set_option('display.max_columns', None)

# ler dataset
df = pd.read_csv('train.csv')
print(df.columns)

# Visualização
lista = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
a = df.pivot_table(index='Sex', columns='Pclass', aggfunc={'Survived': 'mean', 'Fare': 'mean'})
print(a)

age = pd.cut(df['Age'], [0, 20, 30, 40, 50, 60, 70, 80])
pivot = df.pivot_table('Survived', ['Sex', age], 'Pclass')
print(pivot)

piv = df.pivot_table('Survived', 'Sex')
print(piv)