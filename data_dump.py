# here we are going to dump our dataset into mongodb
import pymongo
import pandas as pd
import json

# fetching the data here in the mongodb
client = pymongo.MongoClient(
    "mongodb+srv://NilutpalDAS992:s729TiAxVqw01pG@cluster0.xxk1zfm.mongodb.net/?retryWrites=true&w=majority"
)
db = client.test

# path of data
DATA_FILE_PATH = (
    r"D:\Data Science\Medical insurance\Medical-insurance-prediction\insurance.csv"
)
DATABASE_NAME = "INSURANCE"
COLLECTION_NAME = "INSURANCE_PROJECT"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    df.reset_index(drop=True, inplace=True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
