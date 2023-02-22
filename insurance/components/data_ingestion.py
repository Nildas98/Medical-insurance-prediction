import pandas as pd
import numpy as np
import os, sys
from insurance.entity import config_entity
from insurance.entity import artifact_entity
from insurance.exception import InsuranceException
from insurance import utils
from insurance.logger import logging
from sklearn.model_selection import train_test_split


# Data is divided into train, test and validate
class DataIngestion:
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Export data as pandas Dataframe")
            df: pd.DataFrame = utils.get_collections_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name,
            )
            logging.info(f"Save data in feature store")

            # replacing the null data with NAN
            df.replace(to_replace="na", value=np.NAN, inplace=True)

            logging.info("Create feature store folder if not available")
            feature_store_dir = os.path.dirname(
                self.data_ingestion_config.feature_store_file_path
            )

            # creating a directory for saving the data
            os.makedirs(feature_store_dir, exist_ok=True)
            logging.info("Save df to feature store folder")

            # saving the data
            df.to_csv(
                path_or_buf=self.data_ingestion_config.feature_store_file_path,
                index=False,
                header=True,
            )

            # splitting the data
            logging.info("Splitting the dataset into train and test set")
            train_df, test_df = train_test_split(
                df, test_size=self.data_ingestion_config.test_size, random_state=1
            )

            # creating a dataset directory
            logging.info("Create dataset directory if not exists")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            # saving the train and test data
            logging.info(" Save df to feature store")
            train_df.to_csv(
                path_or_buf=self.data_ingestion_config.train_file_path,
                index=False,
                header=True,
            )
            test_df.to_csv(
                path_or_buf=self.data_ingestion_config.test_file_path,
                index=False,
                header=True,
            )

            # prepare artifact folder
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise InsuranceException(error_message=e, error_detail=sys)
