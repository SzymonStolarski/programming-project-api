from datetime import datetime

from pydantic import BaseModel


class Predictions(BaseModel):

    prediction_id = int
    img = str
    labels = dict
    scores = dict

    class Config:
        orm_mode = True


class Status(BaseModel):

    status_id = int
    prediction_id = int
    status = str
    datetime = datetime

    class Config:
        orm_mode = True


class StatusCreate(BaseModel):

    prediction_id = int
    status = str
    datetime = datetime

    class Config:
        orm_mode = True
