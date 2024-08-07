# -*- coding: utf-8 -*-
"""Missing docstring"""

from .logger import LoggerFactory
from .devices import get_device_info


__version__ = "0.1.0"
__data__ = "2024-08-06"
__author__ = "_NoMem"
__email__ = "novarye.g@gmail.com"
__status__ = "Development"

__all__ = [
    "LoggerFactory",
    "get_device_info",
]

# utils_logger = LoggerFactory(name="utils_logger").logger

# utils_logger.info(f"Module {__name__} loaded")
