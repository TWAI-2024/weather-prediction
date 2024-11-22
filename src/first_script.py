from typing import Tuple, Optional

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from rainpred.data import Dataset

np.random.seed(0)





def main():

    dataset = Dataset()

    X,y = dataset.load_xy()

    # TODO: Include feature engineering
    ### Feature Engineering 

    # features = data.drop(['RainTomorrow', 'Date','day', 'month'], axis=1) # dropping target and extra columns

    # target = data['RainTomorrow']

    # #Set up a standard scaler for the features
    # col_names = list(features.columns)
    # s_scaler = preprocessing.StandardScaler()
    # features = s_scaler.fit_transform(features)
    # features = pd.DataFrame(features, columns=col_names) 



    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



    batch_size = 32
    epochs = 150
    validation_split=0.2
    early_stopping = True

    names = [
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "MLPClassifier"
    ]

    classifiers = [
        DecisionTreeClassifier(max_depth=10, max_features=3, random_state=0),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=3, random_state=0),
        MLPClassifier(alpha=1, max_iter=150, random_state=0, learning_rate_init=0.0001, early_stopping=True, validation_fraction=0.2),
    ]

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        pipe = make_pipeline(StandardScaler(), clf)
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        print(f"Score of {name}:", score)

    # MLPClassifier
    val_acc = clf.validation_scores_
    loss_values = clf.loss_curve_


    # Predicting the test set results
    y_pred = clf.predict(X_test)
    y_pred = (y_pred > 0.5)

    # confusion matrix
    #cmap1 = sns.diverging_palette(260,-10,s=50, l=75, n=5, as_cmap=True)
    #plt.subplots(figsize=(12,8))
    #cf_matrix = confusion_matrix(y_test, y_pred)
    #sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':15})
    print(classification_report(y_test, y_pred))

