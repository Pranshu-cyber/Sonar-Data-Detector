#Rock and mine detection ml program
from pandas import *
from numpy import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm


# Load dataset
dataset=read_csv("sonar data.csv", header=None)


#Seperate dataset in test and train
X=dataset.drop(columns=60, axis=1)
Y=dataset[60]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2, stratify=Y, random_state=2)

#Training Model
model=svm.SVC(kernel='poly')
model.fit(X_train, Y_train)

#Check Accuracy Score
X_train_prediction=model.predict(X_train)
acc1=accuracy_score(X_train_prediction,Y_train)
print('Training data',acc1)
X_test_prediction=model.predict(X_test)
acc2=accuracy_score(X_test_prediction,Y_test)
print('test data:',acc2)

#Running the model
test_data=[0.09, 0.15, 0.23, 0.31, 0.37, 0.42, 0.46, 0.50, 0.52, 0.54, 0.55, 0.56, 0.54, 0.52, 0.50, 0.46, 0.42, 0.38, 0.34, 0.30, 0.26, 0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.018, 0.015, 0.012, 0.010, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.0015, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00008, 0.00006, 0.00004, 0.00002, 0.00001]
test_data=asarray(test_data)
reshaped_data=test_data.reshape(1,-1)
result=model.predict(reshaped_data)
print(result)
