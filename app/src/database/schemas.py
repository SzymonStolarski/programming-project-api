from sqlalchemy import Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from src.database.db import Base


class Predictions(Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, index=True)
    img = Column(String)
    labels = Column(JSONB)
    scores = Column(JSONB)

    # Add relationship
    statuses = relationship("Status", back_populates="prediction")


class Status(Base):
    __tablename__ = "status"

    status_id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.prediction_id"))
    status = Column(String)
    datetime = Column(DateTime)

    # Add relationship
    prediction = relationship("Predictions", back_populates="statuses")
