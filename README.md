# Sensor Fault Detection

A Flask-based machine learning project that predicts whether wafer sensor readings are **Good** or **Bad**. The target convention is:

- `-1` or `0` -> `Bad`
- `1` -> `Good`

The app supports model training, CSV upload prediction, dashboard results, and downloadable prediction output.

## Project Highlights

- End-to-end ML pipeline with ingestion, transformation, training, and prediction layers
- Robust preprocessing for numeric sensor columns
- Handles `-1`, `-1.0`, `0`, `1`, and text labels like `Bad`
- Saves trained model and preprocessor artifacts locally
- Downloads prediction results as `predictions_file.csv`

## Tech Stack

- Python
- Flask
- Pandas, NumPy
- Scikit-learn
- XGBoost
- PyMongo
- HTML/CSS

## Project Structure

```text
src/
  components/
    data_ingestion.py
    data_transformation.py
    model_trainer.py
  pipeline/
    train_pipeline.py
    predict_pipeline.py
  utils/
    main_utils.py
app.py
config/model.yaml
templates/upload_file.html
static/css/style.css
notebooks/wafer_23012020_041211.csv
```

## Run Locally

```powershell
cd "C:\Users\ujjva\OneDrive\Desktop\Sensor Fault"
.\.venv\Scripts\activate
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Routes

- `/` - Dashboard
- `/predict` - GET shows upload page, POST runs prediction
- `/train` - Trains the model and refreshes artifacts
- `/download` - Downloads latest prediction CSV

## Model Output

The prediction output CSV contains the original uploaded sensor columns plus a `Good/Bad` prediction column.

The dashboard shows total processed records, Good/Bad counts, a preview table, and a download button after prediction.

## Artifacts

Runtime artifacts are intentionally ignored by Git:

```text
artifacts/model.pkl
artifacts/preprocessor.pkl
artifacts/wafer_fault.csv
Prediction_Output/
prediction_artifacts/
logs/
```

Train the model locally or place compatible `model.pkl` and `preprocessor.pkl` files inside `artifacts/`.

## Deployment Notes

Before Docker or AWS deployment:

1. Keep secrets in environment variables, not in Git.
2. Train artifacts locally or upload them securely during deployment.
3. Set `PORT` and `HOST` as needed.
