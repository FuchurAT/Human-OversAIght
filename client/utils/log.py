import logging
import os


class Log:
    _logger = None
    _log_file = 'app.log'

    @classmethod
    def set_log_path(cls, log_file):
        """Set the log file path"""
        cls._log_file = log_file

    @classmethod
    def get_logger(cls, name='app', level=logging.INFO):
        if cls._logger is None:
            cls._logger = logging.getLogger(name)
            cls._logger.setLevel(level)

            # Create a file handler for logging
            file_handler = logging.FileHandler(cls._log_file)
            file_handler.setLevel(level)

            # Create a console handler for logging
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # Define a formatter that will be applied to both handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to the logger
            cls._logger.addHandler(file_handler)
            cls._logger.addHandler(console_handler)

        return cls._logger

    @classmethod
    def debug(cls, message):
        cls.get_logger().debug(message)

    @classmethod
    def info(cls, message):
        cls.get_logger().info(message)

    @classmethod
    def warning(cls, message):
        cls.get_logger().warning(message)

    @classmethod
    def error(cls, message):
        cls.get_logger().error(message)

    @classmethod
    def critical(cls, message):
        cls.get_logger().critical(message)


# Example usage:
if __name__ == "__main__":
    Log.info("This is an info message.")
    Log.error("This is an error message.")
