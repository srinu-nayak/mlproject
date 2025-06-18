import pandas as pd

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import sys

@dataclass
class DataTransformationConfig:
    os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_tranformation(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            print(train_data.shape, test_data.shape)


        except Exception as e:
            raise CustomException(e, sys)


