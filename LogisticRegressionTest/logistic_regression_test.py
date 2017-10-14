# Logistic Regression to predict binary outcome

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" Take from our data processing template"""

# Importing the dataset, dataset is a place holder for our data set file
dataset = pd.read_csv('Ads.csv')
#   -1 value is every value up to last column, the -1 values should be set for our data after importing
X = dataset.iloc[:, [2,3]].values
#  -1 in y is to include last column as -1 denotes the last column in the table
y = dataset.iloc[:, 4].values


""" No missing data so this section not required

# Taking care of missing Data
# if we have any missing data it is usual to take the mean of the column with the missing data
from sklearn.preprocessing import Imputer
# sets our imputer up looking for a not a number value NAN and replace it with mean
imputer = Imputer(missing_values="NaN", strategy='mean', axis=0)
# X[:, 1:3] the first : means read all the input records, the 1:3 means check columns 1 and 3 in the test case
imputer = imputer.fit(X[:, 1:3])

"""

""" We do not have any categorical data in this test file so we do not need this section

# Encoding categorical data, we have to do this if we have data that has categories such as
# names, like French, German, England etc.
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
"""

# data set splitting for training and test, 0.25 represents 25%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()