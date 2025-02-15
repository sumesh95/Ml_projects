import numpy as np
import os
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


            reports=evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=regressor_models)

            # Get the model with the highest R² score
            best_model_name = max(reports, key=reports.get)
            best_model_score = regressor_models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with R² Score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model_name)


        except Exception as e:
            pass
