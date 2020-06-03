import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)

#uploading the dataset
data=pd.read_csv('headbrain.csv') #Available in the Github repository

'''print(data.shape)
print(data.head())'''

#assigning the values of X and Y
x=data['Head Size(cm^3)'].values
y=data['Brain Weight(grams)'].values

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

n=len(x)
x=x.reshape((n,1))

reg=LinearRegression()

reg=reg.fit(x,y)

y_pred=reg.predict(x)

r2_score=reg.score(x,y)

print(r2_score)

plt.plot(x,y_pred,color='#58b970',label='Regression Line')

#plotting the points
plt.scatter(x,y_pred,c='#ef5423',label='Scatter Plot')

plt.xlabel('Head Size')
plt.ylabel('Brain Weight')
plt.legend()
plt.show()
