import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Optional
from numpy.typing import ArrayLike


class Visualiser():
    def __init__(self, path: str, name: str, clf: Optional[any], y_test: Optional[ArrayLike], y_pred: Optional[ArrayLike], ):
        self.path = path
        self.name = name
        self.clf = clf
        self.y_test = y_test
        self.y_pred = y_pred
        
    def plot_loss_curve(self):
        try:
            loss_values = self.clf["model"].loss_curve_
            fig, ax = plt.subplots()
            ax.plot(loss_values)
            fig.savefig(self.path + self.name + "_plot_loss_curve.png", dpi=300)
        except AttributeError:
            print('Attribute Error: No loss curve available for this model')
        except TypeError:
            print("There is no loss curve available for this model")


    def plot_validation_scores(self):
        try:
            val_acc = self.clf["model"].validation_scores_
            fig, ax = plt.subplots()
            ax.plot(val_acc)
            fig.savefig(self.path + self.name + "_plot_validation_scores.png", dpi=300)
        except AttributeError:
            print('Attribute Error: No validation scores available for this model')
        except TypeError:
            print('Attribute Error: No validation scores available for this model')



    def plot_conf_matrix(self):
        cf_matrix = confusion_matrix(self.y_test, self.y_pred)
        fig = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
        fig.plot().figure_.savefig(self.path + self.name + "_confusion_matrix.png")


    def visualise_metrics(self):
        self.plot_loss_curve()
        self.plot_validation_scores()
        self.plot_conf_matrix()