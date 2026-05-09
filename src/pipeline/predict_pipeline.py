import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from flask import request
from dataclasses import dataclass
from src.constant import *


@dataclass
class PredictPipelineConfig:
    prediction_output_dirname: str = "Prediction_Output"
    prediction_file_name: str = "predictions_file.csv"
    model_file_path = os.path.join(artifact_folder, "model.pkl")
    preprocessor_file_path = os.path.join(artifact_folder, "preprocessor.pkl")
    prediction_file_path = os.path.join(prediction_output_dirname, prediction_file_name)

class PredictPipeline:
    def __init__(self, request: request):
        self.request = request
        self.predict_pipeline_config = PredictPipelineConfig()
        self.utils = MainUtils()

    def save_input_file(self) -> str:
        try:
           pred_file_input_dir = "prediction_artifacts"
           os.makedirs(pred_file_input_dir, exist_ok=True)
           input_csv_file = self.request.files['file']
           pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
           input_csv_file.save(pred_file_path)
           return pred_file_path
        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, feature: pd.DataFrame):
        try:
            model = self.utils.load_object(self.predict_pipeline_config.model_file_path)
            preprocessor = self.utils.load_object(self.predict_pipeline_config.preprocessor_file_path)
            data_scaled = preprocessor.transform(feature)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


    def get_predicted_dataframe(self, input_dataframe_path: str):
        try:
            prediction_column_name: str = TARGET_COLUMN
            input_dataframe : pd.DataFrame = pd.read_csv(input_dataframe_path)
            input_dataframe = input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe
            feature_dataframe = input_dataframe.drop(columns=TARGET_COLUMN, errors="ignore")
            prediction = self.predict(feature_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in prediction]
            target_column_mapping = {0:'Bad', 1:'Good'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            os.makedirs(self.predict_pipeline_config.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.predict_pipeline_config.prediction_file_path, index=False)
            logging.info("Prediction Completed Successfully! & Prediction file saved at: {}".format(self.predict_pipeline_config.prediction_file_path))
        except Exception as e:
            raise CustomException(e, sys)



    def run_pipeline(self):
        try:
            input_file_path = self.save_input_file()
            self.get_predicted_dataframe(input_file_path)
            return self.predict_pipeline_config.prediction_file_path
        except Exception as e:
            raise CustomException(e, sys)
