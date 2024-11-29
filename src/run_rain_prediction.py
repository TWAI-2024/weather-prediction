import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report

from rainpred.model_factory import ModelFactory
from base.data_base import DataInfo
from base.dataset import Dataset
from utils.load_kaggle_data import load_via_kaggle
from utils.visualisations import Visualiser
from rainpred.data import COL_RAINTOMORROW, RainDataCleaner

from argparse import ArgumentParser

np.random.seed(0)

DATA_PATH = "./data/weatherAUS.csv"

def create_parser():
    parser = ArgumentParser(prog="rainpred")
    parser.add_argument("--download_data", "-s", action='store_true')
    parser.add_argument("--analyze", "-a", action='store_true')
    parser.add_argument("--visualize", "-v", action='store_true')
    parser.add_argument("--result_path", "-r", type=str, default='results/')
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.download_data:
        print("Downloading data...")
        load_via_kaggle('jsphyg/weather-dataset-rattle-package')

    plot_path = args.result_path + "plots/"

    dataset = Dataset(file_path=DATA_PATH, target_col=COL_RAINTOMORROW)

    data = dataset.load_data_frame()

    if args.analyze:
        DataInfo.info(data)
        DataInfo.analyze_unique_values(data)

    # Data Cleaning
    data = RainDataCleaner.clean_data(data)

    X, y = data.drop(columns=COL_RAINTOMORROW), data[COL_RAINTOMORROW]

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

        print(classification_report(y_test, y_pred))

        if args.visualize:
            vis = Visualiser(clf = clf, name = name, path=plot_path, y_test=y_test, y_pred=y_pred)
            vis.visualise_metrics()
        


if __name__ == '__main__':
    main()
