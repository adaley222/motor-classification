"""
helpers.py
----------
Utility functions for the EEG motor imagery pipeline.
"""

import yaml
import logging
from logging.handlers import RotatingFileHandler

def load_config(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_path="logs/app.log", level=logging.INFO):
    """
    Set up logging to file and console with rotation.

    Args:
        log_path (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # File handler with rotation
    fh = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
