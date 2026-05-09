import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from flask import Request
from werkzeug.utils import secure_filename

from src.constant import TARGET_COLUMN, artifact_folder
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class PredictPipelineConfig:
    prediction_input_dirname: str = "prediction_artifacts"
    prediction_output_dirname: str = "Prediction_Output"
    prediction_file_name: str = "predictions_file.csv"
    model_file_path: str = os.path.join(artifact_folder, "model.pkl")
    preprocessor_file_path: str = os.path.join(artifact_folder, "preprocessor.pkl")

    @property
    def prediction_file_path(self) -> str:
        return os.path.join(self.prediction_output_dirname, self.prediction_file_name)


@dataclass
class PredictionFileDetail:
    prediction_file_path: str
    prediction_file_name: str


class PredictionPipeline:
    def __init__(self, request: Request):
        self.request = request
        self.predict_pipeline_config = PredictPipelineConfig()
        self.utils = MainUtils()

    def save_input_file(self) -> str:
        try:
            os.makedirs(self.predict_pipeline_config.prediction_input_dirname, exist_ok=True)
            input_csv_file = self.request.files.get("file")

            if input_csv_file is None or input_csv_file.filename == "":
                raise ValueError("Please upload a CSV file for prediction.")

            if not input_csv_file.filename.lower().endswith(".csv"):
                raise ValueError("Only CSV files are supported.")

            filename = secure_filename(input_csv_file.filename)
            pred_file_path = os.path.join(
                self.predict_pipeline_config.prediction_input_dirname, filename
            )
            input_csv_file.save(pred_file_path)
            return pred_file_path
        except Exception as error:
            raise CustomException(error, sys) from error

    def _validate_model_artifacts(self) -> None:
        missing_paths = [
            file_path
            for file_path in (
                self.predict_pipeline_config.model_file_path,
                self.predict_pipeline_config.preprocessor_file_path,
            )
            if not os.path.exists(file_path)
        ]
        if missing_paths:
            missing_names = ", ".join(os.path.basename(path) for path in missing_paths)
            raise FileNotFoundError(
                f"Missing trained artifact(s): {missing_names}. Train the model before prediction."
            )

    def predict(self, feature: pd.DataFrame):
        try:
            self._validate_model_artifacts()
            model = self.utils.load_object(self.predict_pipeline_config.model_file_path)
            preprocessor = self.utils.load_object(
                self.predict_pipeline_config.preprocessor_file_path
            )
            data_scaled = preprocessor.transform(feature)
            return model.predict(data_scaled)
        except Exception as error:
            raise CustomException(error, sys) from error

    def get_predicted_dataframe(self, input_dataframe_path: str) -> pd.DataFrame:
        try:
            input_dataframe = pd.read_csv(input_dataframe_path)
            input_dataframe = input_dataframe.drop(
                columns=["Unnamed: 0"], errors="ignore"
            )

            feature_dataframe = input_dataframe.drop(columns=TARGET_COLUMN, errors="ignore")
            feature_dataframe = feature_dataframe.select_dtypes(include=[np.number])
            if feature_dataframe.empty:
                raise ValueError("No numeric sensor columns found for prediction.")

            prediction = self.predict(feature_dataframe)

            prediction_column_name = TARGET_COLUMN
            target_column_mapping = {-1: "Bad", 0: "Bad", 1: "Good"}
            input_dataframe[prediction_column_name] = [
                target_column_mapping.get(int(pred), str(pred)) for pred in prediction
            ]

            os.makedirs(self.predict_pipeline_config.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(
                self.predict_pipeline_config.prediction_file_path, index=False
            )
            logging.info(
                "Prediction completed successfully. File saved at: %s",
                self.predict_pipeline_config.prediction_file_path,
            )
            return input_dataframe
        except Exception as error:
            raise CustomException(error, sys) from error

    def run_pipeline(self) -> PredictionFileDetail:
        try:
            input_file_path = self.save_input_file()
            self.get_predicted_dataframe(input_file_path)
            return PredictionFileDetail(
                prediction_file_path=self.predict_pipeline_config.prediction_file_path,
                prediction_file_name=self.predict_pipeline_config.prediction_file_name,
            )
        except Exception as error:
            raise CustomException(error, sys) from error


PredictPipeline = PredictionPipeline
