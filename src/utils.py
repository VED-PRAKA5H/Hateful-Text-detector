import os  # For file/directory operations
import sys  # For system-specific parameters
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Logging setup

here = os.path.dirname(__file__)

def save_object(file_path: str, obj: object) -> None:
    """
    Serialize and save a Python object to a file.

    Parameters:
        file_path (str): Path to save the serialized object
        obj (object): Python object to serialize
    """
    try:
        dir_path = os.path.dirname(file_path)
        # Create directory structure if missing
        os.makedirs(dir_path, exist_ok=True)

        # Serialize and save object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Failed to save object: {e}")
        raise CustomException(e, sys)


def feature_engineering(tweets):
    """
    Perform feature engineering on text data, including tokenization and padding.

    Parameters:
        tweets (list): List of text data

    Returns:
        numpy.ndarray: Matrix of padded sequences
    """
    try:
        logging.info("Starting feature engineering process")
        # Path to the tokenizer file
        tokenizer_file_path = f"{here}/../tokenizer.pkl"

        try:
            # Check if the tokenizer file exists
            if os.path.exists(tokenizer_file_path):
                # Load the tokenizer from the file
                with open(tokenizer_file_path, 'rb') as file:
                    tokenizer = pickle.load(file)
                logging.info("Tokenizer loaded successfully!")
            else:
                # Create a new tokenizer
                max_words = 50000
                tokenizer = Tokenizer(num_words=max_words)
                logging.info("Tokenizer created successfully!")
                # Fit the tokenizer on text data
                tokenizer.fit_on_texts(tweets)
                # Optionally, save the new tokenizer for future use
                with open(tokenizer_file_path, 'wb') as file:
                    pickle.dump(tokenizer, file)
                logging.info("Tokenizer saved successfully!")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Convert text to sequences
        sequences = tokenizer.texts_to_sequences(tweets)

        # Pad sequences to a fixed length
        sequence_matrix = pad_sequences(sequences, maxlen=300)

        logging.info("Feature engineering completed successfully")
        return sequence_matrix

    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        raise CustomException(e, sys)


def evaluate_models(model, X_test, y_test) -> dict:
    """
    Evaluate the given model on test data.

    Parameters:
        model (object): Trained model
        X_test (list): List of test data (text)
        y_test (list): List of true labels for test data
        tokenizer (Tokenizer): Tokenizer used during training

    Returns:
        dict: Evaluation results (accuracy and predictions)
    """
    try:
        logging.info("Starting model evaluation process")

        # Convert test text data to sequences
        test_sequences_matrix = feature_engineering(X_test)

        # Evaluate the model
        acc = model.evaluate(test_sequences_matrix, y_test, verbose=0)

        # Make predictions
        lstm_prediction = model.predict(test_sequences_matrix)

        # Convert predictions to binary labels (threshold = 0.5)
        res = [1 if pred[0] >= 0.5 else 0 for pred in lstm_prediction]

        logging.info("Model evaluation completed successfully")
        return {"accuracy": acc[1], "predictions": res}

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load serialized Python object from file.

    Parameters:
        file_path (str): Path to serialized object file

    Returns:
        object: Deserialized Python object
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Failed to load object: {e}")
        raise CustomException(e, sys)
