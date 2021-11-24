import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as numpy

#Data ingestion and preprocessing
Data= pd.read_table("iris.csv", sep= None, engine= "python")
Data = Data.iloc[0:100,:]
cols= ["Id","SepalLengthCm"]
Data= Data.drop(cols, axis= 1)
data_train, data_test= train_test_split(Data, test_size= 0.5, random_state= 4)
X_train= data_train.drop("Species", axis= 1).values
Y_train= data_train["Species"].values
X_test= data_test.drop("Species", axis=1).values
Y_test= data_test["Species"].values

for i in range(len(Y_train)):
    if (Y_train[i] == "Iris-setosa"):
        Y_train[i] = 0
    elif(Y_train[i] == "Iris-virginica"):
    	Y_train[i] = 2
    else: 
        Y_train[i] = 1


for i in range(len(Y_test)):
    if (Y_test[i] == "Iris-setosa"):
        Y_test[i] = 0
    elif(Y_test[i] == "Iris-virginica"):
    	Y_test[i] = 2
    else:
        Y_test[i] = 1

#Uncomment for MNIST Dataset
# Data = pd.read_table("mnist_train.csv", sep= None, engine= "python")
# X_train= Data.drop("label", axis= 1).values
# Y_train= Data["label"].values

# data_test = pd.read_table("mnist_test.csv", sep= None, engine= "python")
# X_test= data_test.drop("label", axis=1).values
# Y_test= data_test["label"].values

#Feature Scaling
scaler= StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

#K-Means to find cluster centres
cluster_centres = 15
km = KMeans(n_clusters= cluster_centres, max_iter= 100)
km.fit(X_train)
cent = km.cluster_centers_
print(cent)

#Find the maximum distance between two clusters
max=0 
for i in range(cluster_centres):
	for j in range(cluster_centres):
		d= numpy.linalg.norm(cent[i]-cent[j])
		if(d> max):
			max= d
d = max
sigma= d/math.sqrt(2*cluster_centres)

#Calculating phi matrix
shape = X_train.shape
row = shape[0]
column= cluster_centres

Phi = numpy.empty((row,column), dtype= float)
for i in range(row):
    for j in range(column):
        dist= numpy.linalg.norm(X_train[i]-cent[j])
        Phi[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))

#Calucalating the weight matrix for hidden layer -> output layer
PTP = numpy.dot(Phi.T,Phi)
PTP_inv = numpy.linalg.inv(PTP)
fac = numpy.dot(PTP_inv,Phi.T)
W = numpy.dot(fac,Y_train)

#Feed forward with test set
row= X_test.shape[0]
column= cluster_centres
Phi_Test= numpy.empty((row,column), dtype= float)
for i in range(row):
	for j in range(column):
		dist= numpy.linalg.norm(X_test[i]-cent[j])
		Phi_Test[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))

#Final Y_preds
preds = numpy.dot(Phi_Test,W)
preds = abs(preds)
for i in range(preds.shape[0]):
	preds[i] = int(preds[i])
# Y_pred= 0.5*(numpy.sign(numpy.dot(Phi_Test,W)-0.5)+1)
# Y_pred = Y_pred.flatten()
# for i in range(Y_pred.shape[0]):
# 	Y_pred[i] = int(Y_pred[i])
preds = preds.flatten()
preds = preds.tolist()
print(preds)
Y_test = Y_test.tolist()
print(Y_test)
# print(Y_pred)

#Checking Accuracy Score
score = accuracy_score(preds,Y_test)
print (score.mean())