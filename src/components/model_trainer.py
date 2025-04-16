import os
import sys
from dataclasses import dataclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, feature_engineering


@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the model trainer.
    Defines the file path where the trained model will be saved.
    """
    trained_model_file_path = os.path.join("artifacts", "model.h5")  # Path to save the trained model (Keras format)


class ModelTrainer:
    def __init__(self):
        """Initialize the ModelTrainer class and set up configuration."""
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train, test):
        """
        Train and evaluate the LSTM-based model.

        Parameters:
            train (DataFrame): Training dataset.
            test (DataFrame): Testing dataset.

        Returns:
            dict: Evaluation metrics (accuracy, loss, etc.)
        """
        try:
            logging.info("Starting model training process")

            # Split training and testing data
            X_train, y_train = train['tweet'], train['label']
            X_test, y_test = test['tweet'], test['label']

            logging.info("Tokenizing text data")
            max_words = 50000  # Maximum number of unique words to retain

            logging.info("Padding sequences to fixed length")
            maxlen = 300  # Define the fixed sequence length
            train_sequences_matrix = feature_engineering(X_train)
            test_sequences_matrix = feature_engineering(X_test)

            logging.info("Building LSTM model")
            # Define the model
            model = Sequential()
            model.add(Embedding(max_words, 100, input_length=maxlen))  # Embedding layer
            model.add(SpatialDropout1D(0.2))  # Spatial dropout layer
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer
            model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

            logging.info("Training the model")
            history = model.fit(
                train_sequences_matrix,
                y_train,
                batch_size=128,
                epochs=1,  # Set more epochs for better training
                validation_split=0.2
            )

            logging.info("Saving trained model")
            # Save the trained model
            model.save(self.model_trainer_config.trained_model_file_path)

            logging.info("Evaluating the model on test data")
            # Evaluate the model
            evaluation = model.evaluate(test_sequences_matrix, y_test)

            return {
                "loss": evaluation[0],
                "accuracy": evaluation[1],
                "history": history.history  # Include training history
            }

        except Exception as e:
            raise CustomException(e, sys)
