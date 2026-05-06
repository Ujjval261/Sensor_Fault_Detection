import os
import pickle
import sys
from typing import Any

import numpy as np
import yaml

from src.exception import CustomException


def _ensure_parent_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file) or {}
    except Exception as error:
        raise CustomException(error, sys) from error


def write_yaml_file(file_path: str, content: dict, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        _ensure_parent_dir(file_path)

        with open(file_path, "w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(content, yaml_file, sort_keys=False)
    except Exception as error:
        raise CustomException(error, sys) from error


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    try:
        _ensure_parent_dir(file_path)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as error:
        raise CustomException(error, sys) from error


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as error:
        raise CustomException(error, sys) from error


def save_object(file_path: str, obj: Any) -> None:
    try:
        _ensure_parent_dir(file_path)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as error:
        raise CustomException(error, sys) from error


def load_object(file_path: str) -> Any:
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as error:
        raise CustomException(error, sys) from error
