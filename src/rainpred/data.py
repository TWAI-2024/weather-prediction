from utils.data_base import DataInfo, DataTransformer

COL_DATE = "Date"
COL_LOC = "Location"
COL_MINTEMP = "MinTemp"
COL_MAXTEMP = "MaxTemp"
COL_RAINFALL = "Rainfall"
COL_EVAPORATION = "Evaporation"
COL_SUNSHINE = "Sunshine"
COL_WINDGUSTDIR = "WindGustDir"
COL_WINDGUSTSPEED = "WindGustSpeed"
COL_WINDDIR9AM = "WindDir9am"
COL_WINDDIR3PM = "WindDir3pm"
COL_WINDSPEED9AM = "WindSpeed9am"
COL_WINDSPEED3PM = "WindSpeed3pm"
COL_HUMIDITY9AM = "Humidity9am"
COL_HUMIDITY3PM = "Humidity3pm"
COL_PRESSURE9AM = "Pressure9am"
COL_PRESSURE3PM = "Pressure3pm"
COL_CLOUD9AM = "Cloud9am"
COL_CLOUD3PM = "Cloud3pm"
COL_TEMP9AM = "Temp9am"
COL_TEMP3PM = "Temp3pm"
COL_RAINTODAY = "RainToday"
COL_RAINTOMORROW = "RainTomorrow"

class RainDataTransformer(DataTransformer):

    @staticmethod
    def __get_categ_cols(data):
        s = (data.dtypes == "object")
        object_cols = list(s[s].index)
        return object_cols
    
    @staticmethod
    def __get_num_cols(data):
        t = (data.dtypes == "float64")
        num_cols = list(t[t].index)
        return num_cols

    @classmethod
    def clean_data(cls, data):
        # data = super().clean_data(data)
        data = data.drop([COL_DATE], axis=1) # dropping extra columns
        
        # Fill NaNs for categorical columns
        for i in cls.__get_categ_cols(data):
            data[i].fillna(data[i].mode()[0], inplace=True)

        # Fill NaNs for numerical columns
        for i in cls.__get_num_cols(data):
            data[i].fillna(data[i].median(), inplace=True)

        return data
