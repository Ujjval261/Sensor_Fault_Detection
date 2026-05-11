from flask import Flask, render_template, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
import os
import sys
import pandas as pd


from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline


app = Flask(__name__)

MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
WAFER_DATA_PATH = os.path.join("artifacts", "wafer_fault.csv")
PREDICTION_FILE_PATH = os.path.join("Prediction_Output", "predictions_file.csv")


def artifacts_ready():
    return os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)


def render_dashboard(**context):
    base_context = {
        "artifacts_ready": artifacts_ready(),
        "training_data_ready": os.path.exists(WAFER_DATA_PATH),
        "prediction_file_path": None,
        "total_rows": None,
        "good_count": 0,
        "bad_count": 0,
        "preview_table": None,
    }
    base_context.update(context)
    return render_template("upload_file.html", **base_context)


def prediction_context(prediction_file_path):
    prediction_dataframe = pd.read_csv(prediction_file_path)
    prediction_column = "Good/Bad"
    good_count = int((prediction_dataframe[prediction_column] == "Good").sum())
    bad_count = int((prediction_dataframe[prediction_column] == "Bad").sum())
    preview_table = prediction_dataframe.head(10).to_html(
        classes="preview-table", index=False
    )
    return {
        "prediction_file_path": "/download",
        "total_rows": len(prediction_dataframe),
        "good_count": good_count,
        "bad_count": bad_count,
        "preview_table": preview_table,
    }


@app.route("/")
def home():
    return render_dashboard()




@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        model_path = train_pipeline.run_pipeline()
        return render_dashboard(
            success_message=f"Training completed. Model saved at {model_path}"
        )


    except Exception as e:
        return render_dashboard(error_message=str(CustomException(e, sys)))


@app.route('/predict', methods=['POST', 'GET'])
def upload():
   
    try:




        if request.method == 'POST':
            # it is a object of prediction pipeline
            prediction_pipeline = PredictionPipeline(request)
           
            #now we are running this run pipeline method
            prediction_file_detail = prediction_pipeline.run_pipeline()


            lg.info("prediction completed. Showing dashboard result.")
            return render_dashboard(
                success_message="Prediction completed.",
                **prediction_context(prediction_file_detail.prediction_file_path),
            )




        else:
            return render_dashboard()
    except Exception as e:
        return render_dashboard(error_message=str(CustomException(e, sys)))
   

@app.route("/download")
def download_prediction():
    try:
        if not os.path.exists(PREDICTION_FILE_PATH):
            return render_dashboard(error_message="Prediction file not found. Run prediction first.")
        return send_file(
            PREDICTION_FILE_PATH,
            download_name="predictions_file.csv",
            as_attachment=True,
        )
    except Exception as e:
        return render_dashboard(error_message=str(CustomException(e, sys)))





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)
