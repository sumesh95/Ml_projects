import sys
import os
import numpy as np
import dill
from src.logger import logging
from src.exception import CustomeException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


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



def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for name, model in models.items():
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score

        return report

    except Exception as e:
        logging.error(CustomeException(e, sys))
