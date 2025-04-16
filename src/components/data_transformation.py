import sys
from dataclasses import dataclass
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    Defines the file path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts',
                                                   'preprocessor.pkl')  # Path to save the preprocessor object


class DataTransformation:
    def __init__(self):
        """Initialize the DataTransformation class and set up configuration."""
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def get_data_transformer_object(words):
        """
        Method for text cleaning and stemming.
        Parameters:
            words (str): The text to clean and preprocess.
        Returns:
            str: Cleaned and preprocessed text.
        """
        try:
            # Download stopwords if not already downloaded
            nltk.download('stopwords', quiet=True)

            # Initialize stopwords and stemmer
            stopword = set(stopwords.words('english'))
            stemmer = nltk.SnowballStemmer('english')

            words = str(words).lower()  # Convert to lowercase
            words = re.sub(r'\[.*?\]', '', words)  # Remove text in square brackets
            words = re.sub(r'https?://\S+|www\.\S+', '', words)  # Remove URLs
            words = re.sub(r'<.*?>', '', words)  # Remove HTML tags
            words = re.sub(r'[%s]' % re.escape(string.punctuation), '', words)  # Remove punctuation
            words = re.sub(r'\n', '', words)  # Remove newlines
            words = re.sub(r'\w*\d\w*', '', words)  # Remove words containing numbers
            words = re.sub(r'[^\x00-\x7F]+', '', str(words))  # Remove non-ASCII characters
            words = [word for word in words.split() if word not in stopword]  # Remove stopwords
            words = [stemmer.stem(word) for word in words]  # Apply stemming
            words = ' '.join(words)  # Join words back into a single string

            return words

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Method to perform data transformation on training and testing datasets.

        Parameters:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.

        Returns:
            dataframes: Train and Test.
        """
        try:
            # Read training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Specify the target column
            target_column_name = "label"

            # Drop the target column to separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying text preprocessing on training and testing dataframes")

            # Apply text preprocessing using `get_data_transformer_object` method
            input_feature_train_df.apply(self.get_data_transformer_object)
            input_feature_test_df.apply(self.get_data_transformer_object)
            input_feature_train_df['label'] = target_feature_train_df
            input_feature_test_df['label'] = target_feature_test_df
            input_feature_train_df.dropna(inplace=True)
            input_feature_test_df.dropna(inplace=True)
            print(input_feature_train_df.shape, input_feature_test_df.shape)
            logging.info("cleaned train and test dataframes")

            return input_feature_train_df, input_feature_test_df

        except Exception as e:
            raise CustomException(e, sys)

