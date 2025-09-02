# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:03:14 2025

@author: abinnu
"""

import pandas as pd
Data = pd.read_csv("D:/project_classification/dataset/personality_dataset.csv") 

numeric_columns = Data.select_dtypes(include=['int64','float64'])
for column in numeric_columns:
    Data[column] =Data[column].fillna(Data[column].mean())

categorical_column = ["Stage_fear","Drained_after_socializing"]

for column in categorical_column:
    Data[column] = Data[column].fillna(Data[column].mode()[0])
    
Data["Stage_fear"] = Data["Stage_fear"].replace({"Yes":1,"No":0})
Data["Drained_after_socializing"] = Data["Drained_after_socializing"].replace({"Yes":1,"No":0})
Data["Personality"] = Data["Personality"].replace({"Extrovert":1,"Introvert":0})


X = Data.drop("Personality",axis = 1)
Y = Data["Personality"]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

import joblib
joblib.dump(scaler, "D:/project_classification/deployment/ml-personality-app/model/scaler.pkl")


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(scaled_X,Y,test_size=0.2,random_state=0)


from sklearn.neighbors import KNeighborsClassifier as knc

from sklearn.metrics import accuracy_score
acc = []

for i in range(3,30):
    K=knc(n_neighbors = i)
    K.fit(xtrain,ytrain)
    y_pred = K.predict(xtest)
    acc_test = accuracy_score(y_pred,ytest)
    acc.append(acc_test)
acc

k_value = knc(n_neighbors= 9)
y_pred = k_value.fit(xtrain,ytrain).predict(xtest)


path = "D:/project_classification/deployment/ml-personality-app/model/Knnmodel.pkl"
joblib.dump(k_value,path)


from sklearn.metrics import confusion_matrix    

CM = confusion_matrix(ytest,y_pred)    
CM 


Acc = accuracy_score(ytest,y_pred)
accuracy = Acc * 100 
print(f"Accuracy:{accuracy:.2f}%")


from sklearn.metrics import classification_report

CR = classification_report(ytest,y_pred)
CR




