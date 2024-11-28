import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_loss_curve(clf, name, path):
    try:
        loss_values = clf["model"].loss_curve_
        fig, ax = plt.subplots()
        ax.plot(loss_values)
        fig.savefig(path + name + "_plot_loss_curve.png", dpi=300)
    except AttributeError:
        print('Attribute Error: No loss curve available for this model')


def plot_validation_scores(clf, name, path):
    try:
        val_acc = clf["model"].validation_scores_
        fig, ax = plt.subplots()
        ax.plot(val_acc)
        fig.savefig(path + name + "_plot_validation_scores.png", dpi=300)
    except AttributeError:
        print('Attribute Error: No validation scores available for this model')


def plot_conf_matrix(y_test, y_pred, name, path):
    cf_matrix = confusion_matrix(y_test, y_pred)
    fig = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
    fig.plot().figure_.savefig(path + name + "_confusion_matrix.png")


def visualise_metrics(clf, name, path, y_test, y_pred):
    plot_loss_curve(clf, name, path)
    plot_validation_scores(clf, name, path)
    plot_conf_matrix(y_test, y_pred, name, path)