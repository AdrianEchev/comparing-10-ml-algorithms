"""
ANALYSIS OF 10 BINARY CLASSIFICATIO ALGORITHMS
DATASET: Wisconsin Breast Cancer Diagnostics
@author: Adrián Echeverría P.
"""
# OPTIONS OF EXECUTION: 1 IS TO PRINT THE EVALUATION AND 2 IS TO SAVE IT IN A TEXT FILE
OPTION = 2

# imports:
from sklearn.metrics import precision_score, confusion_matrix, f1_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import os
import pandas as pd
from writer import WriteFile
from roc_generator import ROC

# load the datasest:
data_dir = os.path.abspath('../data')
data_path = os.path.join(data_dir, 'data.csv')
df = pd.read_csv(data_path)

# select features and variable of prediction:
df["diagnosis"] = df["diagnosis"].replace({"B": 0, "M": 1})
features = df.loc[:, ["radius_mean", "perimeter_mean", "concave points_mean",
                      "radius_worst", "perimeter_worst", "concave points_worst"]]

X = features
y = df["diagnosis"]

# split test/train data and initialize algorithms in a dictionary:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

classifiers = {
    'LOGISTIC REGRESSION': LogisticRegression(solver='liblinear'),
    'NAIVE BAYES': GaussianNB(),
    'SUPPORT VECTOR MACHINES': SVC(probability=True),
    'DECISION TREE': DecisionTreeClassifier(),
    'RANDOM FOREST': RandomForestClassifier(),
    'GRADIENT BOOSTING MACHINE': GradientBoostingClassifier(),
    'ADABOOST CLASSIFIER': AdaBoostClassifier(),
    'K-NEIGHBORS': KNeighborsClassifier(),
    'NEURAL NETWORK': MLPClassifier(max_iter=1000, solver="adam",),
    'LINEAR DISCRIMINANT ANALYSIS': LinearDiscriminantAnalysis()
}

# run all algorithms and collect their results:
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    cv = cross_val_score(clf, X_train, y_train, cv=5)

    if OPTION == 1:
        print(f"{name}:\n", f" Accuracy: {accuracy}\n", f" Precision: {precision}\n", f" F1 Score: {f1}\n",
              f" Mean Squared Error: {rmse}\n", f" Cross Validation: {cv}\n", f" Confusion Matrix:\n {cm}\n")
        
    elif OPTION ==2:
        WriteFile(name, accuracy, precision, cm, f1, rmse, cv)
        ROC(name, y_test, y_pred_prob)