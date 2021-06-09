from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

data = np.genfromtxt('ibm.us.txt', delimiter=',', skip_header=1, usecols=(1,2,3,4,5,6))
dates = np.genfromtxt('ibm.us.txt', delimiter=',', skip_header=1,usecols=(0), dtype=None)

x = data[:, [0,1,2]] # Open
y = data[:, [3]] # Close
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, stratify=None)
train_dates, test_dates = train_test_split(dates, test_size=0.2, shuffle=False, stratify=None)

#k_range = range(1, 90)
#scores = {}
#scores_list = []

# for k in k_range:
#     knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
#     #Train the model using the training sets
#     knn.fit(X_train, y_train)
#     #Predict the response for test dataset
#     y_pred = knn.predict(X_test)
#     scores[k] = sqrt(mean_squared_error(y_test, y_pred))
#     scores_list.append(sqrt(mean_squared_error(y_test, y_pred)))
#     #print(y_pred)

knn = KNeighborsRegressor(n_neighbors=2, weights="distance")
#Train the model using the training sets
knn.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = knn.predict(X_test)
#print(y_pred)

#plt.plot(k_range, scores_list)
#plt.xlabel("K-Value")
#plt.ylabel("RMSE")
#plt.xticks(np.arange(0, len(x)+1, 1200))
plt.plot(train_dates, y_train)
plt.plot(test_dates, y_test, color="orange", linewidth=1.5)
plt.plot(test_dates, y_pred, color="green", linewidth=0.5)
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.plot()
plt.show()