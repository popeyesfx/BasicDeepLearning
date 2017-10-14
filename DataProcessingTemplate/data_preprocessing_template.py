# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset, dataset is a place holder for our data set file
dataset = pd.read_csv('dataset.csv')
#   -1 value is every value upto last column, the -1 values should be set for our data after importing
X = dataset.iloc[:, :-1].values
#  -1 in y is to include last column as -1 denotes the last column in the table
y = dataset.iloc[:, -1].values

# Taking care of missing Data
# if we have any missing data it is usual to take the mean of the column with the missing data
from sklearn.preprocessing import Imputer
# sets our imputer up looking for a not a number value NAN and replace it with mean
imputer = Imputer(missing_values="NaN", strategy='mean', axis=0)
# X[:, 1:3] the first : means read all the input records, the 1:3 means check columns 1 and 3 in the test case
imputer = imputer.fit(X[:, 1:3])


# Encoding categorical data, we have to do this if we have data that has categories such as
# names, like French, German, Spanish etc.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# 0 is the index of the column with categories
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder gives us dummy records without any weight
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# for dependent variable we use labelencoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# data set splitting for training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


