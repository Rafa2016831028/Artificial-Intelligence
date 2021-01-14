from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
feature_names = boston.feature_names
print(X_train.shape)
print(y_train.shape)
print(feature_names)
print('boston data set object exploring:')
print(dir(boston))

housingDataframe = pd.DataFrame(data= np.c_[boston['data'], boston['target']],
                     columns= np.append(boston['feature_names'], ['target']))

print(housingDataframe.describe())
print(housingDataframe.head(10))

sns.distplot(housingDataframe['target'])

linearRegression = LinearRegression()
linearRegression.fit(X_train,y_train)
print("The R-squared value is: {0:0.4f} \n".format(linearRegression.score(X, y)))
coefficient_df = pd.DataFrame({"Features": ['intercept'] + list(feature_names), "Coef": [model.intercept_] + list(model.coef_)})
print(coefficient_df)
