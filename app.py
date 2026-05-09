import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, send_file

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.constant import TARGET_COLUMN, artifact_folder
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline, PredictPipelineConfig


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

SAMPLE_DATA_PATH = Path("notebooks") / "wafer_23012020_041211.csv"


def model_artifacts_ready() -> bool:
    return all(
        os.path.exists(path)
        for path in (
            os.path.join(artifact_folder, "model.pkl"),
            os.path.join(artifact_folder, "preprocessor.pkl"),
        )
    )


def build_result_context(prediction_file_path: str) -> dict:
    prediction_df = pd.read_csv(prediction_file_path)
    status_counts = prediction_df[TARGET_COLUMN].value_counts().to_dict()
    preview_df = prediction_df.head(8)

    return {
        "prediction_file_path": prediction_file_path,
        "total_rows": len(prediction_df),
        "good_count": int(status_counts.get("Good", 0)),
        "bad_count": int(status_counts.get("Bad", 0)),
        "preview_table": preview_df.to_html(
            classes="preview-table", index=False, border=0
        ),
    }


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "upload_file.html",
        artifacts_ready=model_artifacts_ready(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        predict_pipeline = PredictPipeline(request)
        prediction_file_path = predict_pipeline.run_pipeline()
        context = build_result_context(prediction_file_path)
        return render_template(
            "upload_file.html",
            artifacts_ready=model_artifacts_ready(),
            **context,
        )
    except Exception as error:
        return render_template(
            "upload_file.html",
            artifacts_ready=model_artifacts_ready(),
            error_message=str(CustomException(error, sys)),
        ), 400


@app.route("/train-sample", methods=["POST"])
def train_sample():
    try:
        if not SAMPLE_DATA_PATH.exists():
            raise FileNotFoundError(f"Sample training file not found: {SAMPLE_DATA_PATH}")

        data_transformation = DataTransformation(str(SAMPLE_DATA_PATH))
        train_array, test_array, _ = data_transformation.initiate_data_transformation()
        model_path = ModelTrainer().initiate_model_trainer(train_array, test_array)

        return render_template(
            "upload_file.html",
            artifacts_ready=model_artifacts_ready(),
            success_message=f"Model trained and saved at {model_path}",
        )
    except Exception as error:
        return render_template(
            "upload_file.html",
            artifacts_ready=model_artifacts_ready(),
            error_message=str(CustomException(error, sys)),
        ), 500


@app.route("/download", methods=["GET"])
def download_prediction():
    prediction_file_path = PredictPipelineConfig().prediction_file_path
    if not os.path.exists(prediction_file_path):
        return render_template(
            "upload_file.html",
            artifacts_ready=model_artifacts_ready(),
            error_message="Prediction file is not available yet.",
        ), 404
    return send_file(prediction_file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
