import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainPipelineConfig:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, feature_store_file_path):
        try:
            data_transformation = DataTransformation(feature_store_file_path = feature_store_file_path)
            train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation()
            return train_array, test_array, preprocessor_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, train_array, test_array):
        try:
            model_trainer = ModelTrainer()
            model_report = model_trainer.initiate_model_trainer(train_array, test_array)
            return model_report
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_array, test_array, preprocessor_path = self.start_data_transformation(feature_store_file_path)
            model_report = self.start_model_trainer(train_array, test_array)

            print("Training completed successfully! & Model Report: ", model_report )
        except Exception as e:
            raise CustomException(e, sys)
