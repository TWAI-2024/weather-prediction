from sklearn import linear_model
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from .data import COL_WINDSPEED9AM, COL_CLOUD3PM, COL_CLOUD9AM, COL_EVAPORATION, COL_HUMIDITY3PM, COL_HUMIDITY9AM, COL_LOC, COL_MAXTEMP, COL_MINTEMP, COL_PRESSURE3PM, COL_PRESSURE9AM, COL_RAINFALL, COL_RAINTODAY, COL_SUNSHINE, COL_TEMP3PM, COL_TEMP9AM, COL_WINDDIR3PM, COL_WINDDIR9AM, COL_WINDGUSTDIR, COL_WINDGUSTSPEED, COL_WINDSPEED3PM

class ModelFactory:
    
    @classmethod
    def create_preprocessor(cls):
        return ColumnTransformer(
            [
                ("encoder", OrdinalEncoder(), make_column_selector(dtype_include='object')),
                ("scaler", StandardScaler(),  make_column_selector(dtype_exclude='object'))
            ]
        )
    
    @classmethod
    def create_logistic_regression_orig(cls):
        return Pipeline([
            ("preprocessing", cls.create_preprocessor()),
            ("model", linear_model.LogisticRegression(solver='lbfgs', max_iter=1000))])

    @classmethod
    def create_knn_orig(cls):
        return Pipeline([
            ("preprocessing", cls.create_preprocessor()),
            ("model", KNeighborsClassifier(n_neighbors=1))])

    @classmethod
    def create_random_forest_orig(cls):
        return Pipeline([
            ("preprocessing", cls.create_preprocessor()),
            ("model", RandomForestClassifier(max_depth=10, n_estimators=10, max_features=3, random_state=0))])

    @classmethod
    def create_decision_tree_orig(cls):
        return Pipeline([
            ("preprocessing", cls.create_preprocessor()),
            ("model", DecisionTreeClassifier(max_depth=10, max_features=3, random_state=0))])
    
    @classmethod
    def create_mlp_classifier_orig(cls):
        return Pipeline([
            ("preprocessing", cls.create_preprocessor()),
            ("model", MLPClassifier(alpha=1, max_iter=150, random_state=0, learning_rate_init=0.0001, early_stopping=True, validation_fraction=0.2))])