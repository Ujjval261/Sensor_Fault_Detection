import sys
from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.constant import MONGO_DB_URL
from src.exception import CustomException


class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            local_training_files = [
                Path("artifacts") / "wafer_fault.csv",
                Path("notebooks") / "wafer_23012020_041211.csv",
            ]
            for file_path in local_training_files:
                if file_path.exists():
                    return str(file_path)

            if not MONGO_DB_URL:
                raise ValueError(
                    "MONGO_DB_URL is not set and no local wafer training CSV was found."
                )

            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, feature_store_file_path):
        try:
            data_transformation = DataTransformation(
                feature_store_file_path=feature_store_file_path
            )
            train_array, test_array, preprocessor_path = (
                data_transformation.initiate_data_transformation()
            )
            return train_array, test_array, preprocessor_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, train_array, test_array):
        try:
            model_trainer = ModelTrainer()
            model_path = model_trainer.initiate_model_trainer(train_array, test_array)
            return model_path
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_array, test_array, preprocessor_path = self.start_data_transformation(
                feature_store_file_path
            )
            model_path = self.start_model_trainer(train_array, test_array)

            print("Training completed successfully. Model path:", model_path)
            return model_path
        except Exception as e:
            raise CustomException(e, sys)


TrainPipelineConfig = TrainingPipeline
