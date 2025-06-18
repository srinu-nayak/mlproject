from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.utils import read_sql_data
import os
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # reading the data from mysql
            df = pd.read_csv(os.path.join('notebook/data', 'raw.csv'))
            logging.info('reading raw data completed')

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            df.to_csv(self.config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info('data ingestion completed')

            return (
                self.config.train_data_path,
                self.config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)










