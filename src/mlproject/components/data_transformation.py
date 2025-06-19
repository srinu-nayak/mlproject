import numpy as np
import pandas as pd

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from src.mlproject.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()


    def initiate_preprocessing_steps(self, numerical_features, categorical_features):
        try:
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{num_pipeline}")
            logging.info(f"Numerical Columns:{cat_pipeline}")

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_features),
                ('cat', cat_pipeline, categorical_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_tranformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Training data shape: {}'.format(train_df.shape))
            logging.info('Test data shape: {}'.format(test_df.shape))

            target_column_name = "math_score"

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)

            target_features_train_df = train_df[target_column_name]
            target_features_test_df = test_df[target_column_name]

            numerical_features = input_features_train_df.select_dtypes(include = 'number').columns.tolist()
            categorical_features = input_features_train_df.select_dtypes(exclude = 'number').columns.tolist()

            preprocessing_obj = self.initiate_preprocessing_steps(numerical_features, categorical_features)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_features_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_features_test_df)
            ]

            logging.info('pickle object saved')

            save_object(
                file_path = self.config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )


            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


