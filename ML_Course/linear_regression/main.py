# ML Base Temlate

# base imports
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Const
dataset_loc = 'datasets/linear_regression/Data.csv'
test_size = 1/3
features_num = 3

# import datasets
dataset = pd.read_csv(dataset_loc)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# split data to train test
from sklearn.cross_validation import train_test_split
x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# # feature scaling
# from sklearn.preprocessing import StandaredScaler  # you can use here min max scaler
# sc_X = StandaredScaler()
# x_train = sc_X.fit_transform(x_train)
# x_test = sc_X.transform(x_test)

# from sklearn.linear_model import LinearRegression
# regrissor = LinearRegression()
# regrissor.fit(x_train,x_test)
# y_predict = regrissor.predict(x_test)

# regression algorithm
from sklearn.tree import DecisionTreeRegressor
regrissor = DecisionTreeRegressor(random_state=0)
regrissor.fit(X,y)
y_predict = regrissor.predict(x_test)

# visualize data
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regrissor.predict(x_train),color='blue')
plt.title("salary vs YOE")
plt.xlabel('YOE')
plt.ylabel('salary')
plt.show()
