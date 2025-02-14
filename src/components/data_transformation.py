import sys
import os
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomeException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformerConfig:
    pre_processor_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformer_config=DataTransformerConfig()

    
    def get_data_transformer_object(self):

        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[

                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            logging.info("numerical column transformation is completed")
            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("categorical column transformation is completed")

            pre_processor=ColumnTransformer(

                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)

                ]
            )

            return pre_processor
        
        except Exception as e:
            logging.error(CustomeException(e,sys))


    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df=pd.read_csv(train_path)
            logging.info("training df is loaded")
            test_df=pd.read_csv(test_path)
            logging.info("test df is loaded")

            pre_processing_obj=self.get_data_transformer_object()

            target_column="math_score"

            input_train_df=train_df.drop(columns=[target_column],axis=1)
            target_train_column=train_df[target_column]

            input_test_df=test_df.drop(columns=[target_column],axis=1)
            target_test_column=test_df[target_column]

            input_feature_train_arr=pre_processing_obj.fit_transform(input_train_df)
            input_feature_test_arr=pre_processing_obj.transform(input_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_train_column)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_test_column)]

            save_object(file_path=self.data_transformer_config.pre_processor_file_path,obj=pre_processing_obj)

            return(train_arr,test_arr,self.data_transformer_config.pre_processor_file_path)
        



        except Exception as e:
            logging.error(CustomeException(e,sys))
            pass








    
