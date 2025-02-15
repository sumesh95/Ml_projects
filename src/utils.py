import sys
import os
import numpy as np
import dill
from src.logger import logging
from src.exception import CustomeException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV


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



def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        trained_models = {}  # Store trained models

        for name, model in models.items():
            param_grid = params.get(name, {})  # Fetch params if available

            if param_grid:  # Apply GridSearchCV if params exist
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
                grid_search.fit(x_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                best_model = model
                best_model.fit(x_train, y_train)
                best_params = "Default parameters"

            # Predictions
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            # Model performance
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store results
            report[name] = {
                "train_score": train_model_score,
                "test_score": test_model_score,
                "best_params": best_params
            }

            trained_models[name]=best_model

            # Log details
            logging.info('---------------------------')
            logging.info(f"Model: {name}")
            logging.info(f"Train R² Score: {train_model_score:.4f}")
            logging.info(f"Test R² Score: {test_model_score:.4f}")
            logging.info(f"Best Parameters: {best_params}\n")
            logging.info('---------------------------')

        return report,trained_models

    except Exception as e:
        logging.error(CustomeException(e, sys))
