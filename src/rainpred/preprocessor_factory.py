from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from .data import COL_WINDSPEED9AM, COL_CLOUD3PM, COL_CLOUD9AM, COL_EVAPORATION, COL_HUMIDITY3PM, COL_HUMIDITY9AM, COL_LOC, COL_MAXTEMP, COL_MINTEMP, COL_PRESSURE3PM, COL_PRESSURE9AM, COL_RAINFALL, COL_RAINTODAY, COL_SUNSHINE, COL_TEMP3PM, COL_TEMP9AM, COL_WINDDIR3PM, COL_WINDDIR9AM, COL_WINDGUSTDIR, COL_WINDGUSTSPEED, COL_WINDSPEED3PM

class PreprocessorFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_WINDSPEED9AM, COL_CLOUD3PM, COL_CLOUD9AM, COL_EVAPORATION, COL_HUMIDITY3PM, COL_HUMIDITY9AM, COL_MAXTEMP, COL_MINTEMP, COL_PRESSURE3PM, COL_PRESSURE9AM, COL_RAINFALL, COL_SUNSHINE, COL_TEMP3PM, COL_TEMP9AM, COL_WINDGUSTSPEED, COL_WINDSPEED3PM]


    @classmethod
    def create_ordinal_encoder_standard_scaler(cls):
        return ColumnTransformer(
            [
                ("encoder", OrdinalEncoder(), make_column_selector(dtype_include='object')),
                ("scaler", StandardScaler(),  make_column_selector(dtype_exclude='object'))
            ]
        )
    
    @classmethod
    def create_ordinal_encoder_standard_scaler(cls):
        return ColumnTransformer(
            [
                ("scaler", StandardScaler(),  cls.COLS_USED_BY_ORIGINAL_MODELS)
            ]
        )
