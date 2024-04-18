# ML Base Temlate

# base imports
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Const
dataset_loc = 'datasets/data_preprocessing/Data.csv'
test_size = 0.2
features_num = 3

# import datasets
dataset = pd.read_csv(dataset_loc)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# replace missing values
import sklearn.preprocessing as Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:features_num])
X[:, 1:features_num] = imputer.transform(X[:, 1:features_num])

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# split data to train test
from sklearn.cross_validation import train_test_split

x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# feature scaling
from sklearn.preprocessing import StandaredScaler  # you can use here min max scaler

sc_X = StandaredScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
