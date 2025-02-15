import numpy as np
import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomeException
from src.logger import logging
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifact','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):

        try:
            logging.info('splitting training and test data')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            )
        

            regressor_models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }

            regressor_params = {
                "Linear Regression": {},  # No hyperparameters for linear regression
                "Decision Tree": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 10],
                    "subsample": [0.8, 1.0]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0]
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                },
                "CatBoost": {
                    "iterations": [100, 200, 500],
                    "depth": [4, 6, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "l2_leaf_reg": [1, 3, 5]
                }
            }


            reports,trained_models=evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,
                                   models=regressor_models,params=regressor_params)

            # Get the model with the highest R² score
            best_model_name = max(reports, key=lambda model:reports[model]['test_score'])
            best_model_selected = trained_models[best_model_name]
            best_model_score = reports[best_model_name]

            logging.info(f"Best Model: {best_model_name} with best Score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model_selected)


        except Exception as e:
            logging.error(CustomeException(e,sys))
            
