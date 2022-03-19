from fastapi import FastAPI
import pandas as pd
import uvicorn

from src.image_preprocessing.image_preprocessor import ImagePreprocessor
from src.predictor.predictor import Predictor
from src.data_models.prediction_input import PredictionInput


img_preprocessor = ImagePreprocessor()
efficientdet_lite4_model = Predictor('efficientdet_lite4')
efficientdet_lite2_model = Predictor('efficientdet_lite2')
efficientdet_lite4_model.load_model()
efficientdet_lite2_model.load_model()
model_dictionary = {'EfficientDet-Lite4': efficientdet_lite4_model,
                    'EfficientDet-Lite2': efficientdet_lite2_model}
available_labels = pd.read_csv('src/models/image_labels.csv',
                               sep=';', index_col='ID')['OBJECT (2017 REL.)']

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put("/predict_static_img")
def predict_static_image(input_data: PredictionInput):
    img = img_preprocessor.base64_to_open_cv_img(input_data.img_base64)
    rgb_img, rgb_tensor = img_preprocessor.transform(img)

    selected_model = model_dictionary[input_data.model_name]
    selected_model.predict(rgb_tensor)
    pred_response = selected_model.draw_boxes(
                            rgb_img,
                            min_score=input_data.min_score,
                            filter_predictions=input_data.filter_predictions
                                              )

    return pred_response


@app.get("/available_models")
def get_available_models():
    return {'available_models': list(model_dictionary.keys())}


@app.get("/available_labels")
def get_available_labels():
    return {'available_labels': set(available_labels)}


uvicorn.run(app)
