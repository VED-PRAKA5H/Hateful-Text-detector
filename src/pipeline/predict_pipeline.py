import os
import sys
import pandas as pd
import tensorflow as tf

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.utils import load_object, feature_engineering


def predict(features):
    try:
        cwd = os.path.dirname(__file__)
        model_path = os.path.join(cwd, '..', 'components', 'artifacts', 'model.h5')
        model = tf.keras.models.load_model(model_path)
        data_transformation = DataTransformation()  # Create an instance of DataTransformation class
        features.apply(data_transformation.get_data_transformer_object)
        feature_engineered = feature_engineering(features)
        prediction = model.predict(feature_engineered)
        return prediction

    except Exception as e:
        raise CustomException(e, sys)


class CustomData:
    def __init__(self, tweet: str):
        self.tweet = tweet

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {"tweet": [self.tweet]}
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

