import pandas as pd

dic = {'Nome':['maria','joão','pedro','julia'], 'Idade':['22','10','23','30'], 'peso':[100, 20, 40, 60]}
dic_frame = pd.DataFrame(dic)
print(dic_frame.head())

z = dic_frame.ix[0, :]
y = dic_frame.ix[1, :]

m = pd.DataFrame(z) + pd.DataFrame(y)
print(m)


