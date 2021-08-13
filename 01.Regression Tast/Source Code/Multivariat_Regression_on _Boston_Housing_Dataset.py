# data reading and arrays
import pandas as pd
from sklearn.datasets import load_boston
import numpy as np
from pandas.plotting import scatter_matrix

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import math

# model
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split


boston = load_boston()

"""print(boston.DESCR)"""
# load data indto a pandas data frame
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

""""
data.hist(bins=50,figsize=(30,20))
plt.show() 
"""

#print(data.head())

""" find null values
 for col in data.columns:
        print(col, str(round(100* data[col].isnull().sum() / len(data), 2)) + '%')
    
"""

"""
plotdata=data
plt.figure(figsize=(20, 12))
sns.heatmap(plotdata.corr(), annot=True)
plt.show()
"""

"""
sns.set(style="ticks", color_codes=True)
sns.pairplot(data)
plt.show()
"""
# adding the price column to the dataframe
data['PRICE'] = boston.target

"""
plt.hist(data["PRICE"],bins=10)
plt.show()
"""
# predict attribute
predict = "PRICE"

# create X and Y data
X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])

# for loop for test with random state
#for i in range(10):

# split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#print("Number of training recodes= ",len(x_train))


# create and fit model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# calculate accuracy MSE and RMSE
acc = model.score(x_test, y_test)
mse = sklearn.metrics.mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)



# print accuracy MSE and RMSE
#print("Random State= ",i)
print("Accuracy:", acc*100,"%")
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)

for i in range(len(predictions)):
    print("Actual: ", y_test[i], " Prediction: ", round(predictions[i],1))

'''
# used for the final graph
true_handle = plt.scatter(y_test, y_test, alpha=0.6, color='blue', label='value')
fit = np.poly1d(np.polyfit(y_test,y_test,1))
lims = np.linspace(min(y_test) - 1, max(y_test) + 1)
plt.plot(lims, fit(lims), alpha=0.3, color='black')
pred_handle = plt.scatter(y_test, predictions, alpha=0.6, color='red', label='predicted')
plt.legend(handles=[true_handle,pred_handle], loc='upper left')
plt.show()
'''


