from enum import Enum
import pandas as pd

class FeatureName(Enum):
    YEAR = "year"

class FeatureGenerator():

    @classmethod
    def generate_year(cls, df, ):
        df['Date']= pd.to_datetime(df["Date"])
        #Creating a collumn of year
        df[FeatureName.YEAR] = df.Date.dt.year
        return df