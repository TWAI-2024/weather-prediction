from typing import Tuple
import pandas as pd


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

class Dataset:
    def __init__(self):
        pass
    
    def load_data_frame(self) -> pd.DataFrame:
        data = pd.read_csv("./data/weatherAUS.csv")
        return data
    
    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
        """
        df = self.load_data_frame()
        return df.drop(columns=COL_RAINTOMORROW), df[COL_RAINTOMORROW]