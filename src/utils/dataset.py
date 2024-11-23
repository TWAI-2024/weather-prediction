from typing import Tuple, Optional

import pandas as pd

class Dataset:
    def __init__(self, file_path: str, target_col: str, num_samples: Optional[int] = None):
        self.file_path = file_path
        self.target_col = target_col
        self.num_samples = num_samples  

    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset (including the class column)
        """
        data = pd.read_csv(self.file_path).dropna()
        if self.num_samples is not None:
            df = df.sample(self.num_samples, random_state=self.random_seed)
        return data
    
    def load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        :return: a pair (X, y) where X is the data frame containing all attributes and y is the corresping series of class values
        """
        df = self.load_data_frame()
        return df.drop(columns=self.target_col), df[self.target_col]