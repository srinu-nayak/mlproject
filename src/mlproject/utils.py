import pickle
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import mysql.connector
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.metrics import r2_score

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

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)


    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, model):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(r2_score(y_test, y_pred))

    except Exception as e:
        raise CustomException(e, sys)
