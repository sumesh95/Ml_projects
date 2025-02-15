
from pymongo.mongo_client import MongoClient
import ssl
import certifi
from push_data import ExtractData


if __name__ == '__main__':
    FILE_PATH = 'notebook/data/StudentsPerformance.csv'
    DATABASE = 'StudentPerformanceDataBase'
    COLLECTION = 'StudentsPerformance'
    
    extractDat = ExtractData()  # Ensure ExtractData class is defined correctly
    record = extractDat.cv_to_json_convertor(FILE_PATH)
    no = extractDat.insert_data_into_mongo(record, DATABASE, COLLECTION)
    
    print(no)