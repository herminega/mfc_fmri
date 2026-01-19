# utils/helpers.py
import logging
import os
import yaml

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger that writes to both console and file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
