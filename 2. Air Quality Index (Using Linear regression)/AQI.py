import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import os

missing_values = ["n/a", "na", "--"]
dataSet = pd.read_csv("city_day.csv", na_values=missing_values)
print(dataSet)

#data cleaning
print(dataSet.describe())
print(dataSet.head())
print(dataSet.isnull().sum())

from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
dataSet['city'] = lb_make.fit_transform(dataSet['City'].astype(str))

dataSet.drop('City',axis=1, inplace=True)
dataSet.drop('Date',axis=1, inplace=True)
dataSet.drop('AQI_Bucket',axis=1, inplace=True)

for(columnName, columnData) in dataSet.iteritems():
   median = dataSet[columnName].median()
   dataSet[columnName].fillna(median, inplace=True)

print("After Data cleaning --------------------------")
print(dataSet.isnull().sum())
print(dataSet.columns)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

linear_regression = LinearRegression()
y = dataSet['AQI']
x = dataSet[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
       'Benzene', 'Toluene', 'Xylene','city']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

linear_model = linear_regression.fit(X_train,y_train);
y_prediction = linear_regression.predict(X_test)

print("Predicted Y values : ------------")
print(y_prediction)

plt.scatter(y_test, y_prediction)
plt.xlabel("true values")
plt.ylabel("predictions")

print("Score : ")
print(linear_model.score(X_test, y_test))