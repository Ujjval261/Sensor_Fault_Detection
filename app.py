import os
import sys
import json

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from src.constant import TARGET_COLUMN, artifact_folder
from src.exception import CustomException
from src.logger import logging as lg
from src.pipeline.predict_pipeline import PredictPipelineConfig, PredictionPipeline
from src.pipeline.train_pipeline import TrainingPipeline


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


def model_artifacts_ready():
    return all(
        os.path.exists(path)
        for path in (
            os.path.join(artifact_folder, "model.pkl"),
            os.path.join(artifact_folder, "preprocessor.pkl"),
        )
    )


def load_model_metrics():
    metrics_path = os.path.join(artifact_folder, "model_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def build_prediction_context(prediction_file_path):
    prediction_df = pd.read_csv(prediction_file_path)
    status_counts = prediction_df[TARGET_COLUMN].value_counts().to_dict()
    preview_columns = [column for column in prediction_df.columns if column != TARGET_COLUMN][:6]
    preview_columns.append(TARGET_COLUMN)
    preview_df = prediction_df[preview_columns].head(10)

    return {
        "artifacts_ready": model_artifacts_ready(),
        "model_metrics": load_model_metrics(),
        "prediction_file_path": prediction_file_path,
        "total_rows": len(prediction_df),
        "good_count": int(status_counts.get("Good", 0)),
        "bad_count": int(status_counts.get("Bad", 0)),
        "preview_table": preview_df.to_html(
            classes="preview-table", index=False, border=0
        ),
    }


@app.route("/")
def home():
    return render_template(
        "upload_file.html",
        artifacts_ready=model_artifacts_ready(),
        model_metrics=load_model_metrics(),
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        model_path = train_pipeline.run_pipeline()
        lg.info("Training completed. Model saved at %s", model_path)
        return render_template(
            "upload_file.html",
            artifacts_ready=model_artifacts_ready(),
            model_metrics=load_model_metrics(),
            success_message=f"Training completed. Model saved at {model_path}",
        )
    except Exception as e:
        raise CustomException(e, sys)


@app.route("/predict", methods=["POST", "GET"])
def upload():
    try:
        if request.method == "POST":
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("Prediction completed. Rendering prediction summary.")
            return render_template(
                "upload_file.html",
                **build_prediction_context(prediction_file_detail.prediction_file_path),
            )

        return render_template(
            "upload_file.html",
            artifacts_ready=model_artifacts_ready(),
            model_metrics=load_model_metrics(),
        )
    except Exception as e:
        raise CustomException(e, sys)


@app.route("/download")
def download_prediction():
    config = PredictPipelineConfig()
    if not os.path.exists(config.prediction_file_path):
        return render_template(
            "upload_file.html",
            artifacts_ready=model_artifacts_ready(),
            model_metrics=load_model_metrics(),
            error_message="Prediction file is not available yet. Run prediction first.",
        ), 404

    return send_file(
        config.prediction_file_path,
        download_name=config.prediction_file_name,
        as_attachment=True,
    )


@app.route("/metrics")
def metrics():
    model_metrics = load_model_metrics()
    if model_metrics is None:
        return jsonify({"error": "Model metrics are not available. Train the model first."}), 404
    return jsonify(model_metrics)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "127.0.0.1")
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
