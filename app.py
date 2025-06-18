from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig


if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion().initiate_data_ingestion()

    except Exception as e:
        logging.info('custom exception occurred')
        raise CustomException(e, sys)

