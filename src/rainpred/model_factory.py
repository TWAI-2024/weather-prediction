from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.rainpred.preprocessor_factory import PreprocessorFactory

class ModelFactory:
    
    @classmethod
    def create_logistic_regression_orig(cls):
        return Pipeline([
            ("preprocessing", PreprocessorFactory.create_ordinal_encoder_standard_scaler()),
            ("model", linear_model.LogisticRegression(solver='lbfgs', max_iter=1000))])

    @classmethod
    def create_knn_orig(cls):
        return Pipeline([
            ("preprocessing", PreprocessorFactory.create_ordinal_encoder_standard_scaler()),
            ("model", KNeighborsClassifier(n_neighbors=1))])

    @classmethod
    def create_random_forest_orig(cls):
        return Pipeline([
            ("preprocessing", PreprocessorFactory.create_ordinal_encoder_standard_scaler()),
            ("model", RandomForestClassifier(max_depth=10, n_estimators=10, max_features=3, random_state=0))])

    @classmethod
    def create_decision_tree_orig(cls):
        return Pipeline([
            ("preprocessing", PreprocessorFactory.create_ordinal_encoder_standard_scaler()),
            ("model", DecisionTreeClassifier(max_depth=10, max_features=3, random_state=0))])
    
    @classmethod
    def create_mlp_classifier_orig(cls):
        return Pipeline([
            ("preprocessing", PreprocessorFactory.create_ordinal_encoder_standard_scaler()),
            ("model", MLPClassifier(alpha=1, max_iter=150, random_state=0, learning_rate_init=0.0001, early_stopping=True, validation_fraction=0.2))])