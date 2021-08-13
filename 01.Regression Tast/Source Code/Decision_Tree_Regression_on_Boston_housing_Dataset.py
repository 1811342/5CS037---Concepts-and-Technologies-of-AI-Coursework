# data reading and arrays
import pandas as pd
from sklearn.datasets import load_boston
import numpy as np

# for plotting
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import math

# model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor



boston = load_boston()
# load data indto a pandas data frame
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

# adding the price column to the dataframe
data['PRICE'] = boston.target

# predict attribute
predict = "PRICE"

# create X and Y data
X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])

# for loop for test with random state
#for i in range(10):

# split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=9)
#print("Number of training recodes= ",len(x_train))

# create and fit model
model = DecisionTreeRegressor(random_state=0)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# calculate accuracy MSE and RMSE
acc = model.score(x_test, y_test)
mse = sklearn.metrics.mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)

# print accuracy MSE and RMSE
#print("Random State= ", i)

print("Accuracy:", acc*100,"%")
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)

for i in range(len(predictions)):
    print("Actual: ", y_test[i], " Prediction: ", round(predictions[i],1))

'''
true_handle = plt.scatter(y_test, y_test, alpha=0.6, color='blue', label='value')
fit = np.poly1d(np.polyfit(y_test,y_test,1))
lims = np.linspace(min(y_test) - 1, max(y_test) + 1)
plt.plot(lims, fit(lims), alpha=0.3, color='black')
pred_handle = plt.scatter(y_test, predictions, alpha=0.6, color='red', label='predicted')
plt.legend(handles=[true_handle,pred_handle], loc='upper left')
plt.show()
'''

'''
# creating decision tree.dot
export_graphviz(model, out_file="tree.dot",rounded=True,filled=True)
'''



