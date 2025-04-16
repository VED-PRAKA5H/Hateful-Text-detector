import sys  # Module for system-specific parameters and functions


def error_message_details(error, error_detail: sys):
    """
    Generate a detailed error message including the file name, line number, and error message.

    Parameters:
        error (Exception): The exception object that contains the error information.
        error_detail (sys): The sys module used to extract traceback details.

    Returns:
        str: A formatted error message string.
    """

    _, _, exc_tb = error_detail.exc_info()  # Extract traceback information from the exception
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the name of the file where the error occurred

    # Format the error message to include file name, line number, and the actual error message
    error_message = "Error Occurred in Python Script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message  # Return the formatted error message


class CustomException(Exception):
    """
    Custom exception class that extends the built-in Exception class.
    This class is used to raise exceptions with detailed error messages.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the CustomException with an error message and details.
        Parameters:
            error_message : The message describing the error.
            error_detail (sys): The sys module used to extract traceback details.
        """

        super().__init__(error_message)  # Call the base class constructor with the error message
        self.error_message = error_message_details(error_message,
                                                   error_detail=error_detail)  # Generate detailed error message

    def __str__(self):
        """Return the string representation of the CustomException."""
        return self.error_message  # Return the detailed error message when the exception is printed
