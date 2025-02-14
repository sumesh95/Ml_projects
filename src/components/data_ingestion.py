import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifact','train.csv')
    test_data_path=os.path.join('artifact','test.csv')
    raw_data_path=os.path.join('artifact','raw.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        
        try:
            df=pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Entered the data ingestion method')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            logging.info('create directory for train test and raw')
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_data,test_data=train_test_split(df,test_size=0.25,random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion completed')
            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)



        except Exception as e:
            logging.error(CustomeException(e,sys))







