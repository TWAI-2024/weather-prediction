import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report

from rainpred.model_factory import ModelFactory
from utils.data_base import DataInfo
from utils.dataset import Dataset
from utils.load_kaggle_data import load_via_kaggle
from rainpred.data import COL_RAINTOMORROW, RainDataCleaner

np.random.seed(0)

def main():
    load_via_kaggle('jsphyg/weather-dataset-rattle-package')

    dataset = Dataset(file_path="./data/weatherAUS.csv", target_col=COL_RAINTOMORROW)

    data = dataset.load_data_frame()

    DataInfo.info(data)
    DataInfo.analyze_unique_values(data)

    # Data Cleaning

    data = RainDataCleaner.clean_data(data)

    X, y = data.drop(columns=COL_RAINTOMORROW), data[COL_RAINTOMORROW]

    # TODO: Include feature engineering
    # Feature Engineering 

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    names = [
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "MLPClassifier"
    ]

    classifiers = [
        ModelFactory.create_decision_tree_orig(),
        ModelFactory.create_random_forest_orig(),
        ModelFactory.create_mlp_classifier_orig()
    ]

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f"Score of {name}:", score)
        # Predicting the test set results
        y_pred = clf.predict(X_test)
        # y_pred = (y_pred > 0.5)
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
