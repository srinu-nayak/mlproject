import pickle
import sys

from sklearn.model_selection import GridSearchCV

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

def evaluate_models(X_train, y_train, X_test, y_test, models:dict, params):
    global r2_score
    try:
        results = {}
        trained_models = {}
        best_params = {}

        for model_name, model in models.items():

            param_grid = params.get(model_name, {})
            # print(param_grid)

            gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2')
            gs.fit(X_train, y_train)

            # Predict using the best estimator
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)

            test_r2 = r2_score(y_test, y_pred)

            results[model_name] = test_r2
            trained_models[model_name] = best_model
            best_params[model_name] = gs.best_params_

        return results, trained_models, best_params

    except Exception as e:
        raise CustomException(e, sys)

# def save_object(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)
#         os.makedirs(dir_path, exist_ok=True)
#         with open(file_path, 'wb') as f:
#             pickle.dump(obj, f)
#     except Exception as e:
#         raise CustomException(e, sys)