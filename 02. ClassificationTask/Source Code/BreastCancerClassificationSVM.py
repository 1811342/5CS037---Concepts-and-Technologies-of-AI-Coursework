""""
Brest Cancer classification with SVM
author :- Nirmal Mudiyanselage 1811342
5CS037
Last edited 20-10-2020
"""
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm
import seaborn as sns


# load CSV data
data = pd.read_csv("Breast Cancer.csv")
# print(data.head())

"""
# test - how number of train data affect the accuracy
data=data.loc[1:50, :]
print('num of training recodes ',data.shape)
"""

# convert non numerical values to numerical values
tr = preprocessing.LabelEncoder()
diagnosis = tr.fit_transform(list(data["diagnosis"]))
data["diagnosis"] = diagnosis
data = data.drop(["Unnamed: 32"], axis=1)

# print(data.head())

arrtibutes = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
              "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean"
, "fractal_dimension_mean"]

'''
#scatter plot
scatter_matrix(data[arrtibutes],figsize=(20,20))
plt.show()
'''

'''
#heatmap
plotdata=data[arrtibutes]
plt.figure(figsize=(20, 12))
sns.heatmap(plotdata.corr(), annot=True)
plt.show()
'''

'''

# find null values
for col in data.columns:
    print(col, str(round(100 * data[col].isnull().sum() / len(data), 2)) + '%')
# hist 
data.hist(bins=50,figsize=(30,20))
plt.show()
'''

X = list(zip(data["radius_mean"], data["texture_mean"], data["perimeter_mean"], data["area_mean"],
             data["smoothness_mean"], data["compactness_mean"], data["concavity_mean"], data["concave points_mean"],
             data["symmetry_mean"], data["fractal_dimension_mean"]))

Y = list(diagnosis)

# split tast and train data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=9)


# build model
model = svm.SVC(kernel="linear", C=3)
model.fit(x_train, y_train)

# predict
prediction = model.predict(x_test)

# print ACC
acc = model.score(x_test, y_test)
MSE = sklearn.metrics.mean_squared_error(y_test, prediction)
rmse = math.sqrt(MSE)

# plot confusion matrix
matrix = plot_confusion_matrix(model,x_test,y_test,cmap=plt.cm.Blues,normalize="true")
plt.title("Confusion matrix")
#plt.show(matrix)
plt.show()

# print("Random State= ",i)

print("Accuracy:", acc * 100, "%")

# inverse transform data tp print
y_test = tr.inverse_transform(y_test)
prediction = tr.inverse_transform(prediction)

# display results
for i in range(len(prediction)):
    print("Actual: ", y_test[i], " Prediction: ", prediction[i])

