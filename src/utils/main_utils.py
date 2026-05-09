import importlib
import os
import pickle
import sys
from typing import Any

import numpy as np

try:
    yaml = importlib.import_module("yaml")
except ImportError as error:
    raise ModuleNotFoundError(
        "PyYAML is required for MainUtils. Install it with `pip install PyYAML`."
    ) from error

from src.exception import CustomException
from src.logger import logging


def _ensure_parent_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


class MainUtils:
    def read_yaml_file(self, file_path: str) -> dict:
        try:
            with open(file_path, "r", encoding="utf-8") as yaml_file:
                return yaml.safe_load(yaml_file) or {}
        except Exception as error:
            raise CustomException(error, sys) from error

    def read_schema_config_file(self) -> dict:
        try:
            return self.read_yaml_file(os.path.join("config", "schema.yaml"))
        except Exception as error:
            raise CustomException(error, sys) from error

    @staticmethod
    def write_yaml_file(file_path: str, content: dict, replace: bool = False) -> None:
        try:
            if replace and os.path.exists(file_path):
                os.remove(file_path)

            _ensure_parent_dir(file_path)

            with open(file_path, "w", encoding="utf-8") as yaml_file:
                yaml.safe_dump(content, yaml_file, sort_keys=False)
        except Exception as error:
            raise CustomException(error, sys) from error

    @staticmethod
    def save_object(file_path: str, obj: Any) -> None:
        logging.info("Entered the save_object method of MainUtils class")

        try:
            _ensure_parent_dir(file_path)

            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

            logging.info("Exited the save_object method of MainUtils class")
        except Exception as error:
            raise CustomException(error, sys) from error

    @staticmethod
    def load_object(file_path: str) -> Any:
        logging.info("Entered the load_object method of MainUtils class")

        try:
            with open(file_path, "rb") as file_obj:
                loaded_object = pickle.load(file_obj)

            logging.info("Exited the load_object method of MainUtils class")
            return loaded_object
        except Exception as error:
            raise CustomException(error, sys) from error

    @staticmethod
    def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
        try:
            _ensure_parent_dir(file_path)

            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
        except Exception as error:
            raise CustomException(error, sys) from error

    @staticmethod
    def load_numpy_array_data(file_path: str) -> np.ndarray:
        try:
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj, allow_pickle=True)
        except Exception as error:
            raise CustomException(error, sys) from error
