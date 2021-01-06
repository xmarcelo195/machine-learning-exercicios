import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('winequality_clean.csv')
corr_matrix = df.corr()

print(corr_matrix)

'''
alcohol
- volatile_acidity
sulphates
Citric_acid
'''