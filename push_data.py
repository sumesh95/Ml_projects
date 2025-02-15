import os
import sys
import json
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient

load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

import certifi
import numpy as np
import pandas as pd
import pymongo




class ExtractData:
    def __init__(self):
        pass
    
    
    
    def cv_to_json_convertor(self,file_path):
        data=pd.read_csv(file_path)
        data.reset_index(drop=True)
        records=list(json.loads(data.T.to_json()).values())
        return records
    
    def insert_data_into_mongo(self,records,database,collection):
        self.database=database
        self.collection=collection
        self.records=records
        self.mongo_client=MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
        self.database=self.mongo_client[self.database]
        self.collection=self.database[self.collection]
        self.collection.insert_many(self.records)
        print(f"Successfully inserted {len(self.records)} documents into {database}.{collection}")
        
        
        
