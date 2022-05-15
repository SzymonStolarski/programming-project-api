from datetime import datetime

from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
import uvicorn
import pandas as pd
from sqlalchemy.orm import Session

from src.data_models import user_inputs
from src.database.db import SessionLocal, engine
from src.database import crud, schemas
from src.image_preprocessing.image_preprocessor import ImagePreprocessor
from src.predictor.predictor import Predictor


# Load object detection
img_preprocessor = ImagePreprocessor()
efficientdet_lite4_model = Predictor('efficientdet_lite4')
efficientdet_lite2_model = Predictor('efficientdet_lite2')
efficientdet_lite4_model.load_model()
efficientdet_lite2_model.load_model()
model_dictionary = {'EfficientDet-Lite4': efficientdet_lite4_model,
                    'EfficientDet-Lite2': efficientdet_lite2_model}
available_labels = pd.read_csv('src/models/image_labels.csv',
                               sep=';', index_col='ID')['OBJECT (2017 REL.)']

schemas.Base.metadata.create_all(bind=engine)

app = FastAPI()


def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict_static_img/")
def predict_static_image(input_data: user_inputs.PredictionInput,
                         background_tasks: BackgroundTasks,
                         db: Session = Depends(get_db)):

    # Create empty Prediction table record
    prediction_record = crud.create_prediction_record(db)
    # Create entry - to start
    crud.create_status_by_prediction_id(
        db, input={'prediction_id': prediction_record.prediction_id,
                   'status': 'to_start',
                   'datetime': datetime.now()})
    # Run the predictions in background
    background_tasks.add_task(_run_predictions, db, input_data,
                              prediction_record.prediction_id)

    return {'prediction_id': prediction_record.prediction_id}


def _run_predictions(db: Session, input_data: user_inputs.PredictionInput,
                     prediction_id: int) -> None:
    # Create entry in db - running
    crud.create_status_by_prediction_id(db, input={
        'prediction_id': prediction_id,
        'status': 'running',
        'datetime': datetime.now()})

    try:
        img = img_preprocessor.base64_to_open_cv_img(input_data.img_base64)
        rgb_img, rgb_tensor = img_preprocessor.transform(img)
        selected_model = model_dictionary[input_data.model_name]
        selected_model.predict(rgb_tensor)
        pred_response = selected_model.draw_boxes(
                                rgb_img,
                                min_score=input_data.min_score,
                                filter_predictions=input_data
                                                    .filter_predictions
                                                )
        # Save results to db
        crud.update_prediction_record(db, input={
                'prediction_id': prediction_id,
                'img': pred_response['img'],
                'labels': pred_response['labels'],
                'scores': pred_response['scores']})
        # Update entry - finished
        crud.create_status_by_prediction_id(db, input={
            'prediction_id': prediction_id,
            'status': 'finished',
            'datetime': datetime.now()})
        print('Prediction finished and db updated')

    except BaseException as e:
        crud.create_status_by_prediction_id(db, input={
            'prediction_id': prediction_id,
            'status': 'failed',
            'datetime': datetime.now()})
        raise e('Error during model predictions')


@app.get("/predictions/{prediction_id}")
def read_prediction(prediction_id: int, db: Session = Depends(get_db)):

    prediction = crud.get_prediction(db, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404,
                            detail="Prediction with given ID not found")

    # Temporary workaround, as the `response_model` in constructor
    # gives some problems.
    prediction_response = prediction.__dict__

    return prediction_response


@app.get("/prediction_statuses/{prediction_id}")
def read_prediction_statuses(prediction_id: int,
                             db: Session = Depends(get_db)):
    prediction_statuses = crud.get_prediction_statuses(db, prediction_id)
    if len(prediction_statuses) == 0:
        raise HTTPException(status_code=404,
                            detail="No statuses for this prediction_id")

    # Temporary workaround, as the `response_model` in constructor
    # gives some problems.
    prediction_statuses_response = {k: v.__dict__ for (k, v)
                                    in enumerate(prediction_statuses)}

    return prediction_statuses_response


@app.get("/available_models")
def get_available_models():
    return {'available_models': list(model_dictionary.keys())}


@app.get("/available_labels")
def get_available_labels():
    return {'available_labels': set(available_labels)}


uvicorn.run(app)
