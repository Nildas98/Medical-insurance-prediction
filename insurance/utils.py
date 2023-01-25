import pandas as pd
import numpy as np
import os, sys
from insurance.exception import InsuranceException
from insurance.config import mongo_client
from insurance.logger import logging
import yaml
import numpy as np
import dill

# defining a function so that we can store our data locally
def get_collections_as_dataframe(
    database_name: str, collection_name: str
) -> pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(
            f"Reading data from database: {database_name} and collection {collection_name}"
        )
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"find columns in the data{df.columns}")
        if "_id" in df.columns:
            logging.info(f"dropping columns: _id")
            df = df.drop("_id", axis=1)
        logging.info(f"Rows and Columns in df: {df.shape}")
        return df

    except Exception as e:
        raise InsuranceException(e, sys)
