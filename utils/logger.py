# -*- encoding: utf-8 -*-

"""
This module provides a factory class for creating and configuring a logger with both
console and file output capabilities. It uses the `logging` module for standard logging
and `colorlog` for colored console output.

The `LoggerFactory` class allows for flexible logger configuration:
- Set the logger's name and logging level.
- Optionally configure the logger to output to a file.
- Customize console log color formatting.

Usage:
    To use the `LoggerFactory`, instantiate it with the desired configuration. The
    logger is automatically created during initialization and can be accessed via
    the `logger` attribute.

Example:
    factory = LoggerFactory(name="utils_logger", to_file=True, file_path="example.log")
    logger = factory.logger
    logger.info("This is an info message")
    logger.error("This is an error message")
"""

import sys
import os
import logging
import colorlog

__all__ = ["LoggerFactory"]


class LoggerFactory:
    """
    A factory class for creating and configuring a logger with console and file output capabilities.

    The LoggerFactory allows you to:
    - Specify the logger's name and logging level.
    - Optionally configure logging to a file.
    - Customize the color formatting for console output.

    Attributes:
        name (str): The name of the logger.
        level (int): The logging level for the logger (default is logging.INFO).
        to_file (bool): Whether to save logs to a file (default is False).
        file_path (str): The file path for the log file; applicable only if to_file is True (default is "app.log").
        logger (logging.Logger): The configured logger instance created during initialization.

    Methods:
        create_logger() -> logging.Logger:
            Create a logger with the specified configuration.
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        to_file: bool = False,
        file_path: str = "app.log",
    ):
        """
        Initialize the LoggerFactory and create a logger.

        :param name: The name of the logger.
        :param level: The logging level of the logger (default is logging.INFO).
        :param to_file: Whether to save logs to a file (default is False).
        :param file_path: The file path for the log file; applicable only if to_file is True (default is "app.log").
        """
        self.name = name
        self.level = level
        self.to_file = to_file
        self.file_path = file_path
        self.logger = self.create_logger()  # Automatically create the logger

    def create_logger(self) -> logging.Logger:
        """
        Create a logger with the specified configuration.

        This method configures the logger according to the settings defined during
        initialization. It sets up console output with color formatting and, if specified,
        adds a file handler to log messages to a file.

        :return: A configured logger instance.
        """
        # Create a logger
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)

        # Custom color configuration
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_formatter)

        # Add the console handler to the logger
        if not logger.hasHandlers():  # Avoid adding multiple handlers
            logger.addHandler(console_handler)

        # Add a file handler if to_file is True
        if self.to_file:
            # Create the log file if it does not exist
            file_handler = logging.FileHandler(self.file_path)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(file_handler)

        return logger


if __name__ == "__main__":
    # Create a LoggerFactory instance and get the logger
    logger = LoggerFactory(
        name="utils_logger", to_file=True, file_path="example.log"
    ).logger
    logger.info("This is an info message")
    logger.error("This is an error message")
