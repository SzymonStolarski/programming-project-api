from sqlalchemy.orm import Session

from src.data_models import models
from src.database import schemas


def create_prediction_record(db: Session) -> schemas.Predictions:
    """
    Create empty record in the predictions table.

    Parameters
    -----------
    db : sqlalchemy.orm.Session
        SqlAlchemy session object.

    Returns
    -------
    db_predictions : schemas.Predictions
        SqlAlchemy object containing the prediction record.
    """
    db_predictions = schemas.Predictions()
    db.add(db_predictions)
    db.commit()
    db.refresh(db_predictions)

    return db_predictions


def update_prediction_record(db: Session,
                             input: models.Predictions) -> schemas.Predictions:
    """
    Update empty prediction record with values from objectd detection
    predictions.

    Parameters
    -----------
    db : sqlalchemy.orm.Session
        SqlAlchemy session object.
    input : models.Predictions
        Input with values that will update the record.

    Returns
    -------
    predictions_record : schemas.Predictions
        SqlAlchemy object containing the prediction record.
    """
    # First get the record
    prediction_record = db.query(schemas.Predictions).filter(
        schemas.Predictions.prediction_id == input['prediction_id']).first()

    # Update the record with prediction results
    prediction_record.img = input['img']
    prediction_record.labels = input['labels']
    prediction_record.scores = input['scores']
    db.commit()
    db.refresh(prediction_record)

    return prediction_record


def create_status_by_prediction_id(db: Session,
                                   input: models.StatusCreate
                                   ) -> schemas.Status:
    """
    Create record in the status table.

    Parameters
    -----------
    db : sqlalchemy.orm.Session
        SqlAlchemy session object.
    input : models.StatusCreate
        Input with values that will create the record.

    Returns
    -------
    db_status : schemas.Predictions
        SqlAlchemy object containing the status record.
    """
    db_status = schemas.Status(prediction_id=input['prediction_id'],
                               status=input['status'],
                               datetime=input['datetime'])
    db.add(db_status)
    db.commit()
    db.refresh(db_status)

    return db_status


def get_prediction(db: Session, id: int) -> schemas.Predictions:
    """Get prediction SqlAlchemy object by prediction_id."""
    return db.query(schemas.Predictions).filter(
        schemas.Predictions.prediction_id == id).first()


def get_prediction_statuses(db: Session, id: int) -> list[schemas.Status]:
    """Get status SqlAlchemy object by prediction_id."""
    return db.query(schemas.Status).filter(
        schemas.Status.prediction_id == id).all()
