#Impoorting the package
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
dataset= pd.read_csv(r'C:\Users\kumar.sanu\Desktop\ML\Data_Preprocessing\Data.csv')
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,:-3].values

#Taking care of missing data
"""from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])"""

#Encoding Categorical data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X=LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder=OneHotEncoder(categorical_features=[0])
X=oneHotEncoder.fit_transform(X).toarray()
labelEncoder_y=LabelEncoder()
y=labelEncoder_X.fit_transform(y)"""

#Split the dataset into Test and Training Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""








