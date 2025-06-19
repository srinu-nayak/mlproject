from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression

from src.mlproject.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Initiating model trainer')
            X_train, y_train, X_test, y_test = train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]

            model = LinearRegression()

            evaluate_models (
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model= model
            )






        except Exception as e:
            raise CustomException(e, sys)
