import sys
import os
import numpy as np
import dill
from src.logger import logging
from src.exception import CustomeException


def save_object(file_path, obj):
    try:
        # Extract directory from the file path and create it if needed
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save object as a pickle file
        with open(file_path, "wb") as file:
            dill.dump(obj, file)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        logging.error(CustomeException(e, sys))