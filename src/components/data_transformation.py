import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from src.exception import CustomException
from src.logger import logging
from src.constant import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir= os.path.join(artifact_folder)
    transformed_train_file_path = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path = os.path.join(artifact_dir, 'test.npy')
    transformed_object_file_path = os.path.join(artifact_dir, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self, feature_store_file_path: str):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)
            if TARGET_COLUMN not in data.columns:
                raise ValueError(f"Target column '{TARGET_COLUMN}' not found in input data")
            return data
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def get_feature_target_dataframe(dataframe: pd.DataFrame):
        try:
            features = dataframe.drop(columns=TARGET_COLUMN)
            features = features.drop(columns=["Unnamed: 0"], errors="ignore")
            features = features.select_dtypes(include=[np.number])

            if features.empty:
                raise ValueError("No numeric sensor columns found for model training.")

            target_values = dataframe[TARGET_COLUMN].astype(str).str.strip().str.lower()
            target = np.where(target_values.isin(["-1", "0", "bad"]), 0, 1)
            return features, target
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_objects(self):
        try:
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())
            preprocessor = Pipeline(steps=[imputer_step, scaler_step])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self):
        logging.info("Entered the initiate_data_transformation method of the Data_Transformation class")
        try:
            dataframe  = self.get_data(feature_store_file_path=self.feature_store_file_path)
            X, y = self.get_feature_target_dataframe(dataframe)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            preprocessor = self.get_data_transformer_objects()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            preprocessor_file_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_file_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_file_path, obj=preprocessor)
            train_arr= np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]
            return (train_arr, test_arr, preprocessor_file_path)
        except Exception as e:
            raise CustomException(e, sys)
