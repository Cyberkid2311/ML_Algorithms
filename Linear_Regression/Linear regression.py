import os
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

#Calculating the mean using numpy
mean_x=np.mean(x)
mean_y=np.mean(y)

n=len(x) #Count of the values

#Calculating B1 and B0
#y = (B1*x)+ B0
numerator = 0
denominator = 0
for i in range(n):
    numerator+= (x[i]-mean_x)*(y[i]-mean_y)
    denominator+=(x[i]-mean_x)**2
b1=numerator/denominator
b0=mean_y - (b1*mean_x)

'''print(b1,b0)'''

#Plotting the Values and the Regression line on the graph
max_x= np.max(x)+100
min_x= np.min(x)-100

#Calculating the Regression line values of X and Y
X= np.linspace(min_x,max_x,1000)
Y=b0+(b1*X)

#Plotting the line
plt.plot(X,Y,color='#58b970',label='Regression Line')

#plotting the points
plt.scatter(x,y,c='#ef5423',label='Scatter Plot')

plt.xlabel('Head Size')
plt.ylabel('Brain Weight')
plt.legend()
plt.show()


#Calculating the r-squared value
unexpected_variation = 0
total_variation = 0
for i in range(n):
    y_predicted= b0+ b1*x[i]
    total_variation += (y[i]-mean_y)**2
    unexpected_variation += (y[i]-y_predicted)**2

r_square = 1- (unexpected_variation/total_variation)
print(r_square)
