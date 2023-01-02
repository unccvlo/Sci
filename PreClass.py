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

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
