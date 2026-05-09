# Sensor Fault Detection

A Flask-based machine learning project that predicts whether wafer sensor readings are **Good** or **Bad**. The target convention is:

- `-1` or `0` -> `Bad`
- `1` -> `Good`

The app supports model training, CSV upload prediction, a prediction summary page, downloadable prediction output, and a metrics endpoint.

## Project Highlights

- End-to-end ML pipeline with ingestion, transformation, training, and prediction layers
- Robust preprocessing for numeric sensor columns
- Handles `-1`, `-1.0`, `0`, `1`, and text labels like `Bad`
- Saves trained model and preprocessor artifacts locally
- Shows prediction counts and preview table in the UI
- Downloads prediction results as `predictions_file.csv`
- Saves model metrics in `artifacts/model_metrics.json`
- Includes health and metrics API routes for deployment readiness

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

- `/` - Upload and prediction UI
- `/predict` - GET shows upload page, POST runs prediction
- `/download` - Downloads latest prediction CSV
- `/train` - Trains the model and refreshes artifacts
- `/metrics` - Returns model metrics as JSON
- `/health` - Returns app health status

## Model Output

The prediction output CSV contains the original uploaded sensor columns plus a `Good/Bad` prediction column.

The UI shows:

- total processed records
- Good count
- Bad count
- preview table
- download button

## Artifacts

Runtime artifacts are intentionally ignored by Git:

```text
artifacts/model.pkl
artifacts/preprocessor.pkl
artifacts/model_metrics.json
Prediction_Output/
prediction_artifacts/
logs/
```

Train the model locally or place compatible `model.pkl` and `preprocessor.pkl` files inside `artifacts/`.

## Deployment Notes

Before Docker or AWS deployment:

1. Keep secrets in environment variables, not in Git.
2. Use `/health` for load balancer health checks.
3. Train artifacts locally or upload them securely during deployment.
4. Set `PORT` and `HOST` as needed.
