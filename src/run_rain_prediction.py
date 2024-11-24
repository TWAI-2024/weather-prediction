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
from rainpred.data import COL_RAINTOMORROW, RainDataTransformer

np.random.seed(0)

def main():
    load_via_kaggle('jsphyg/weather-dataset-rattle-package')

    dataset = Dataset(file_path="./data/weatherAUS.csv", target_col=COL_RAINTOMORROW)

    data = dataset.load_data_frame()

    DataInfo.info(data)

    DataInfo.analyze_unique_values(data)

    # Data Cleaning

    data = RainDataTransformer.clean_data(data)

    X, y = data.drop(columns=COL_RAINTOMORROW), data[COL_RAINTOMORROW]
    
    # TODO: Include feature engineering
    # Feature Engineering 

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
        #y_pred = (y_pred > 0.5)

        # confusion matrix
        #cmap1 = sns.diverging_palette(260,-10,s=50, l=75, n=5, as_cmap=True)
        #plt.subplots(figsize=(12,8))
        #cf_matrix = confusion_matrix(y_test, y_pred)
        #sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':15})
        print(classification_report(y_test, y_pred))

    # MLPClassifier
    val_acc = clf["model"].validation_scores_
    loss_values = clf["model"].loss_curve_
    #print("val_acc:", val_acc)
    #print("loss_values:", loss_values)

if __name__ == '__main__':
    main()
