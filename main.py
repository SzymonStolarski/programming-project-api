from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.image_preprocessing.image_preprocessor import ImagePreprocessor
from src.predictor.predictor import Predictor


class ModelInput(BaseModel):
    img_base64: str
    model_name: str
    min_score: Optional[float] = 0.4
    filter_predictions: Optional[list[str]] = None


app = FastAPI()

img_preprocessor = ImagePreprocessor()
efficientdet_lite4_model = Predictor('efficientdet_lite4').load_model()
efficientdet_lite2_model = Predictor('efficientdet_lite2').load_model()

model_dictionary = {'efficientdet_lite4': efficientdet_lite4_model,
                    'efficientdet_lite2': efficientdet_lite2_model}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.put("/predict_static_img")
def predict_static_image(input_data: ModelInput):
    img = img_preprocessor.base64_to_open_cv_img(input_data.img_base64)
    rgb_img, rgb_tensor = img_preprocessor.transform(img)

    selected_model = model_dictionary[input_data.model_name]
    selected_model.predict(rgb_tensor)
    pred_response = selected_model.draw_boxes(rgb_img,
                                         min_score=input_data.min_score,
                                         filter_predictions=input_data.filter_predictions
                                        )
    
    return pred_response
