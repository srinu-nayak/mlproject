from altair import data_transformers

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation, DataTransformationConfig
from src.mlproject.components.model_tranier import ModelTrainerConfig, ModelTrainer

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_tranformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)


    except Exception as e:
        logging.info('custom exception occurred')
        raise CustomException(e, sys)



