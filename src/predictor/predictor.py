import base64
from time import time

import pandas as pd
import numpy as np
import tensorflow
import tensorflow_hub as hub
import cv2

from src.exceptions.exceptions import *


MODEL_DICTIONARY = {
    'efficientdet_lite2': 'src/models/efficientdet_lite2_detection_1',
    'efficientdet_lite4': 'src/models/efficientdet_lite4_detection_2'}


class Predictor:
    """
    Prediction class that stores the tensorflow object detection model,
    runs the predictions on a tensor object and draws the boxes with
    the predictions on an image.

    For the purpose of this project, models are downloaded from TensorFlow Hub
    website (see reference) and stored locally in src/models.

    Parameters
    -----------
    model name : str
        Name of the model, that corresponds to the one of the keys
        in MODEL_DICTIONARY, which stores the paths to given model.
        Current list of locally available models:
            - efficentdet_lite2,
            - efficentdet_lite4.

    References
    -----------
    https://tfhub.dev/s?module-type=image-object-detection

    Example
    -----------
    >>> predictor = Predictor('efficientdet_lite2')
    >>> predictor.load_model()
    starting to load efficientdet_lite2 model...
    efficientdet_lite2 model loaded!
    >>> predictor.predict(rgb_tensor)
    >>> image_with_boxes = predictor.draw_boxes(rgb_image, min_score=0.2,
                                            filter_predictions=['person'])
    """
    def __init__(self, model_name: str) -> None:
        self.__model_name = model_name
        self.__check_if_model_included()
        self.__is_model_loaded = False
        self.__is_prediction_done = False

    def load_model(self) -> None:
        """
        Function that loads the selected TensorFlow object detection model
        specified in __init__.
        """
        # Loading model - for this project's purpose storing models locally
        print(f'starting to load {self.__model_name} model...')
        self.model = hub.load(MODEL_DICTIONARY[self.__model_name])

        # Loading csv with labels of classes
        labels = pd.read_csv('src/models/image_labels.csv',
                             sep=';', index_col='ID')
        self.labels = labels['OBJECT (2017 REL.)']

        self.__is_model_loaded = True
        print(f'{self.__model_name} model loaded!')

    def predict(self,
                tensor: tensorflow.int32) -> None:
        """
        Function that makes prediction on given tensor.

        Parameters
        -----------
        tensor : tensorflow.python.python.framework.ops.EagerTensor
            TensorFlow tensor object that stores the image.
        """
        self.__check_is_model_loaded()
        # Creating prediction and measure time
        start_measuring_time = time()
        boxes, scores, classes, _ = self.model(tensor)
        self.__prediction_time = round(time() - start_measuring_time, 2)
        print(f'Prediction time: {self.__prediction_time} seconds')

        # Processing outputs - they will be stored in given instance
        # in order not to rerun the model every time we want to visualize
        # different results on the image
        predicted_labels = classes.numpy().astype('int')[0]
        self.__predicted_labels = np.array([self.labels[i]
                                            for i in predicted_labels])
        self.__predicted_boxes = boxes.numpy()[0].astype('int')
        self.__predicted_scores = scores.numpy()[0]

        self.__is_prediction_done = True

    def draw_boxes(self, img: np.ndarray, min_score: float = 0.4,
                   filter_predictions: list[str] = None) -> np.ndarray:
        """
        Function which draws boxes of predictions on an image.

        Parameters
        -----------
        img : np.ndarray
            Image on which the boxes should be drawn.
        min_score : float
            Minimum prediction confidence level of the drawn objects.
        filter_predictions : list[str], optional
            List of labels that should be drawn only, example:
            filter_predictions = ['person']

        Returns
        ----------
        np.ndarray
            Image containing the prediction boxes.
        """
        self.__check_is_prediction_done()
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_boxes = bgr.copy()

        self.scores_to_visualize = self.__predicted_scores
        self.boxes_to_visualize = self.__predicted_boxes
        self.labels_to_visualize = self.__predicted_labels

        # Filtering if filter_predictions provided - drawing only
        # selected labels
        if filter_predictions is not None:
            # Select only those indexes that have selected labels provided
            # in filter_predictions
            idxes = [idx for idx, label in enumerate(self.__predicted_labels)
                     if label in filter_predictions]
            self.labels_to_visualize = np.array([self.__predicted_labels[i]
                                            for i in idxes])
            self.boxes_to_visualize = np.array([self.predicted_boxes[i]
                                           for i in idxes])
            self.scores_to_visualize = np.array([self.__predicted_scores[i]
                                            for i in idxes])

        drawn_objects_counter = 0
        labels_dict = {}
        scores_dict = {}
        boxes_dict = {}
        # Putting the boxes and labels on the image
        for score, (ymin, xmin, ymax, xmax), label in zip(
                                                    self.scores_to_visualize,
                                                    self.boxes_to_visualize,
                                                    self.labels_to_visualize
                                                        ):
            # Filtering based on the min_score
            if score < min_score:
                continue

            score_txt = f'{100 * round(score,2)}'
            img_boxes = cv2.rectangle(img_boxes, (xmin, ymax), (xmax, ymin),
                                      (255, 0, 255), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_boxes, f'{drawn_objects_counter}: {label}',
                        (xmin, ymax-10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(img_boxes, score_txt, (xmax, ymax-10), font, 0.5,
            #             (255, 0, 0), 1, cv2.LINE_AA)

            labels_dict[drawn_objects_counter] = label
            scores_dict[drawn_objects_counter] = score
            boxes_dict[drawn_objects_counter] = ({'ymin': ymin, 'xmin': xmin,
                               'ymax': ymax, 'xmax': xmax})
            drawn_objects_counter += 1
        print(f'Number of objects drawn: {drawn_objects_counter}')

        img_boxes_rgb = img_boxes[:,:,::-1]
        img_boxes_str = base64.b64encode(cv2.imencode('.jpg', img_boxes_rgb)[1]).decode()

        prediction_response = {
            'img': img_boxes_str,
            'labels': labels_dict,
            'scores': scores_dict,
            'boxes': boxes_dict
        }

        return prediction_response

    def __check_if_model_included(self) -> None:
        if self.__model_name not in MODEL_DICTIONARY.keys():
            raise ModelNotIncludedError(
                f"Selected model provided in model_name not available "
                f"locally. Available models: {list(MODEL_DICTIONARY.keys())}"
            )

    def __check_is_model_loaded(self) -> None:
        if not self.__is_model_loaded:
            raise NotLoadedModelError(
                f"This instance of {self.__class__.__name__} has not "
                f"a loaded model yet; please call `load_model` first."
            )

    def __check_is_prediction_done(self) -> None:
        if not self.__is_prediction_done:
            raise PredictionsNotDoneError(
                f"This instance of {self.__class__.__name__} has not "
                f"yet made predictions with loaded model; "
                f"please call `predict` first."
            )

    @property
    def model_name(self) -> str:
        """str: Returns the model_name"""
        return self.__model_name

    @model_name.setter
    def model_name(self, value) -> None:
        self.__model_name = value

    @property
    def is_model_loaded(self) -> bool:
        """
        bool: Information if the TensforFlow object detection
            model has been loaded to the Predictor instance.
        """
        return self.__is_model_loaded

    @property
    def is_prediction_done(self) -> bool:
        """
        bool: Information if the Predictor instance has made
            predictions on a tensor.
        """
        return self.__is_prediction_done

    @property
    def predicted_labels(self) -> np.ndarray:
        """np.ndarray: Array of prediction labels."""
        return self.__predicted_labels

    @property
    def predicted_boxes(self) -> np.ndarray:
        """np.ndarray: Array of prediction boxes coordinates."""
        return self.__predicted_boxes

    @property
    def predicted_scores(self) -> np.ndarray:
        """np.ndarray: Array of predicted scores."""
        return self.__predicted_scores
