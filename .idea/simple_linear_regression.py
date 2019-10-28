import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r'C:\Users\kumar.sanu\Desktop\ML\Simple_Linear_Regression\Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Train_Test_Split
from sklearn.model_selection import train_test_split
X_train,X_Test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#Fitting the Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test Set Results
y_Pred=regressor.predict(X_Test)

#Visualising the training set
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience in ML(Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set
plt.scatter(X_Test,y_test, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience in ML(Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


