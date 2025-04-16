import logging  # Import the logging module for logging messages
import os  # Module for interacting with the operating system
from datetime import datetime  # Import datetime class for timestamping log files

# Generate a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path for the logs directory, which is located in the same directory as this script
logs_path = os.path.join(os.path.dirname(__file__), "logs")

# Create the logs directory if it does not already exist
os.makedirs(logs_path, exist_ok=True)

# Define the complete path for the log file by joining the logs directory and log file name
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the file where logs will be saved
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define the format of log messages
    level=logging.INFO  # Set the logging level to INFO; this means all messages at this level and above will be logged
)
