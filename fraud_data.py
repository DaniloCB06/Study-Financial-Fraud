import pandas as pd
import numpy as np
import random as rnd
import pickle
from google.colab import drive
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPRegressor
from sklearn import tree, preprocessing
from sklearn.metrics import  confusion_matrix,accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from google.colab import drive

drive.mount('/content/drive')

data.info()
display(data)
abc = data.groupby('type')[['type']].count()
display(abc)

data1 = data.sort_values(by = 'isFraud',ascending = True)
display(data1)

fraude = data.groupby('isFraud')[['isFraud']].count()
display(fraude)

lista_index = []
for i in range(6362620):
  lista_index.append(i)
data1['indice'] = lista_index
data1.set_index('indice',inplace = True)
display(data1)

lista = []
for n in range(6346407):
  lista.append(n)
  
j = 6362619
while j > 6362406:
  j = j - 1
  lista.append(j)


label_encoder = preprocessing.LabelEncoder()
data1['type']= label_encoder.fit_transform(data1['type'])
data1['type']

data_new = data1.drop(['isFlaggedFraud'],axis=1,inplace=False)
uso = data_new.loc[lista]
data_new = data1.drop(lista)
data_new = data_new.reset_index().sample(frac=1)
data_new.drop("indice", inplace = True, axis = 1)
display(data_new)

fraude2 = data_new.groupby('isFraud')[['isFraud']].count()
display(fraude2)

data_uso = uso.iloc[6346257:6346557]
data_uso = data_uso.reset_index().sample(frac=1)

fraud = data_uso.groupby('isFraud')[['isFraud']].count()
display(fraud)
data_uso = data_uso.drop(["nameOrig", "nameDest"], axis = 1)
data_new = data_new.drop(["nameOrig", "nameDest"], axis = 1)
data_uso.drop("indice", inplace = True, axis = 1)
display(data_uso)

data_new=data_new.sample(frac=1).reset_index(drop=True)

X = data_new.drop(["isFraud"],axis=1)
y = data_new.isFraud

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1, shuffle= True)
X_train.shape,X_test.shape

clf = tree.DecisionTreeClassifier(random_state = 1, criterion = "entropy" )
clf = clf.fit(X_train, y_train)
y_pre = clf.predict(X_test)
print(accuracy_score(y_test, y_pre))
print(classification_report(y_test, y_pre))

clf2 = MLPClassifier(random_state = 1,hidden_layer_sizes = (100, 100), activation = 'relu', solver = 'adam',learning_rate_init = 0.003, max_iter=1000).fit(X_train, y_train)
y_pre = clf2.predict(X_test)
print(accuracy_score(y_test, y_pre))
print(classification_report(y_test, y_pre))

clf3 = RandomForestClassifier(n_estimators = 500,random_state=1, criterion = 'entropy')
clf3 = clf3.fit(X_train, y_train)
y_pre = clf3.predict(X_test)
print(accuracy_score(y_test, y_pre))
print(classification_report(y_test, y_pre))

arquivo = open('/content/drive/MyDrive/rede_neural.pkl', 'wb')
pickle.dump(clf2, arquivo)
arquivo.close()
arquivo = open('/content/drive/MyDrive/dados_treino_teste', 'wb')
pickle.dump(data_new, arquivo)
arquivo.close()
arquivo = open('/content/drive/MyDrive/dados_uso', 'wb')
pickle.dump(data_uso, arquivo)
arquivo.close()

#dps do flask

pd.read_pickle('/content/drive/MyDrive/dados_uso')
