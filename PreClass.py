import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn


df = pd.read_csv('/Users/sonveni/Documents/PREP/mushrooms.csv')
# print(df.head()) # prints preview of csv
# print(df.columns)  # prints columns of csv
# print(df.shape)  # returns number of rows x columns
# print(df.dtypes)  # returns the data types of each column
# print(type(df['class'][0]))  # returns the first row data type
# print(df.describe())  # returns description of data
# print(df.isna().sum())  # checks columns for null values

# sns.countplot() # initialize countplot
# sns.set_theme(palette='Accent') # change color theme
# ax = sns.countplot(x="class", data=df) # set x to class column and data to the df
# plt.show() # show the count plot


# print(df['class'].value_counts())  # show the count of values in class column

# ax = sns.countplot(x='bruises', data=df, hue='class')  # plot bar graph with x
# value as bruises and hue as class column
# plt.show()  # show plot

# from sklearn.preprocessing import LabelEncoder
# enc = LabelEncoder()  # initialize label encoder as enc

# for col in df.columns:  # label encode these columns to make them numerical
    # df[col] = enc.fit(df[col]).transform(df[col])

# print(df.dtypes)
# Notice how they have all been transformed to integers now.
# This may allow us to work with more visualizations that require integers.
# However, remember that these are still categorical values.
# A value of 5 is not necessarily "greater than" a value of "2"
# (again, since they are still just different categories).

x = df.drop(['class'], axis = 1)  # use all columns except 'class'
y = df['class']  # target value

# print(x.shape, y.shape)
from sklearn.model_selection import train_test_split  # used to train and test dataset for modeling
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.80)
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier
print(model.fit(x, y))  # this trains the model

from sklearn.metrics import accuracy_score
y_predictions = model.predict(x)
accuracy_score(y_predictions, y)

