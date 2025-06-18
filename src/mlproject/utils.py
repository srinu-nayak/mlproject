import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import mysql.connector
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
database = os.getenv('database')


def read_sql_data():
    logging.info('reading sql data')

    try:
        mydb = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        print("Database connection successful!")

        df = pd.read_sql_query("SELECT * FROM students", mydb)
        # print(df.head(5))
        return df

    except CustomException as e:
        raise CustomException(e, sys)