class NotLoadedModelError(ValueError, AttributeError):
    """
    Exception class to raise if model is used before loading.
    """


class PredictionsNotDoneError(ValueError, AttributeError):
    """
    Exception class to raise if boxes are drawn before predictions.
    """


class ModelNotIncludedError(ValueError, AttributeError):
    """
    Exception class to raise if selected model not available locally.
    """
