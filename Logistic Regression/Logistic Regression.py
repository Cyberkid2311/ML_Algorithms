import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


cancer=pd.read_csv('data.csv')

#print(cancer.head())

cancer=cancer.drop(['id','Unnamed: 32'],axis=1)


def encoding(cancer):
    le=LabelEncoder()
    for col in cancer.columns:
        if cancer[col].dtypes == 'object':
            cancer[col]=le.fit_transform(cancer[col])
    return cancer


encoded_cancer = encoding(cancer)

#print(encoded_cancer.head())

Y= cancer.diagnosis
X= cancer.drop('diagnosis',axis=1)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state=42)

lr = LogisticRegression(max_iter=5000)
lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)
acc1 =accuracy_score(y_test,y_predict)
print(acc1*100)
mat1 = confusion_matrix(y_predict,y_test)
print(mat1)
