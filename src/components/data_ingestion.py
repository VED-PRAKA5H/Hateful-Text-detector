import os  # Module for interacting with the operating system
import sys  # Module for system-specific parameters and functions
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Logging module for tracking events
import pandas as pd  # Library for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Function to split datasets into training and testing sets
from dataclasses import dataclass  # Decorator for creating data classes
from src.components.data_transformation import DataTransformation  # Data transformation component
from src.components.model_trainer import ModelTrainer  # Model training component


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    Uses a dataclass to define paths for train, test, and raw data.
    """
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Path to save training data
    test_data_path: str = os.path.join('artifacts', 'test.csv')  # Path to save testing data
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')  # Path to save raw data


class DataIngestion:
    def __init__(self):
        """Initialize the DataIngestion class and set up configuration."""
        self.ingestion_config = DataIngestionConfig()  # Create an instance of the configuration class

    def initiate_data_ingestion(self):
        """Method to perform data ingestion."""
        logging.info("Entered the data ingestion method")  # Log entry into the method
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv('../../notebook/data/processed.csv')
            logging.info('Read the dataset as dataframe')  # Log successful reading of the dataset

            # Create directories if they do not exist for saving files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Log initiation of train-test split

            # Split the dataset into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the testing set to CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")  # Log completion of data ingestion

            return (
                self.ingestion_config.train_data_path,  # Return path to training data
                self.ingestion_config.test_data_path  # Return path to testing data
            )

        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if an error occurs


if __name__ == "__main__":
    obj = DataIngestion()  # Create an instance of DataIngestion class
    train_data, test_data = obj.initiate_data_ingestion()  # Initiate data ingestion process

    data_transformation = DataTransformation()  # Create an instance of DataTransformation class
    clean_train, clean_test = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()  # Create an instance of ModelTrainer class
    acc = model_trainer.initiate_model_trainer(clean_train, clean_test)  # Train the model and get accuracy
    logging.info(f"Evaluation completed, accuracy: {acc['accuracy']}")

