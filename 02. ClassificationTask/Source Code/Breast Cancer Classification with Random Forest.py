import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# load CSV data
data = pd.read_csv("Breast Cancer.csv")

# convert non numerical values to numerical values
tr = preprocessing.LabelEncoder()
diagnosis = tr.fit_transform(list(data["diagnosis"]))
data["diagnosis"] = diagnosis
data = data.drop(["Unnamed: 32"], axis=1)

# features we are going to use for this model
arrtibutes = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
              "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean"
, "fractal_dimension_mean"]

X = list(zip(data["radius_mean"], data["texture_mean"], data["perimeter_mean"], data["area_mean"],
             data["smoothness_mean"], data["compactness_mean"], data["concavity_mean"], data["concave points_mean"],
             data["symmetry_mean"], data["fractal_dimension_mean"]))

Y = list(diagnosis)

# split tast and train data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# build model
model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(x_train,y_train)

# predict
prediction = model.predict(x_test)

# print ACC
acc = model.score(x_test, y_test)
print("Accuracy: ", acc*100, "%")

# plot confusion matrix
matrix = plot_confusion_matrix(model, x_test, y_test, cmap=plt.cm.Blues,normalize="true")
plt.title("Confusion matrix")
#plt.show(matrix)
plt.show()

# inverse transform data tp print
y_test = tr.inverse_transform(y_test)
prediction = tr.inverse_transform(prediction)

# display results
for i in range(len(prediction)):
    print("Actual: ", y_test[i], " Prediction: ", prediction[i])
