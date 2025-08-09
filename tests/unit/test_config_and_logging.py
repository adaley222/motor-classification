"""
test_config_and_logging.py
-------------------------
Test config loading and logging setup.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.helpers import load_config, setup_logging

def test_load_config():
    config = load_config("config/config.yaml")
    assert isinstance(config, dict)
    assert "input_shape" in config
    assert "n_classes" in config

def test_setup_logging(tmp_path):
    log_path = tmp_path / "test.log"
    setup_logging(str(log_path))
    import logging
    logger = logging.getLogger()
    logger.info("Test log entry")
    assert os.path.exists(log_path)
    with open(log_path, "r") as f:
        log_content = f.read()
    assert "Test log entry" in log_content
