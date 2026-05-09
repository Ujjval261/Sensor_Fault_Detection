import os
import sys

from flask import Flask, jsonify, render_template, request, send_file

from src.exception import CustomException
from src.logger import logging as lg
from src.pipeline.predict_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainingPipeline


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.route("/")
def home():
    return "Welcome to my application"


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        model_path = train_pipeline.run_pipeline()
        lg.info("Training completed. Model saved at %s", model_path)
        return "Training Completed."
    except Exception as e:
        raise CustomException(e, sys)


@app.route("/predict", methods=["POST", "GET"])
def upload():
    try:
        if request.method == "POST":
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("Prediction completed. Downloading prediction file.")
            return send_file(
                prediction_file_detail.prediction_file_path,
                download_name=prediction_file_detail.prediction_file_name,
                as_attachment=True,
            )

        return render_template("upload_file.html")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
