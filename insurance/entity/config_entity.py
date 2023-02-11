import os, sys
from insurance.exception import InsuranceException
from insurance.logger import logging
from datetime import datetime

FILE_NAME = "insurance.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(
                os.getcwd(), "artifact", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}"
            )
        except Exception as e:
            raise InsuranceException(e, sys)


# DataIngestion first read the data and divides it into training, testing and validation file
class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            # calling the database
            self.database_name = "INSURANCE"
            self.collection_name = "INSURANCE_PROJECT"
            # calling the data
            self.data_ingestion_dir = os.path.join(
                training_pipeline_config.artifact_dir, "data_ingestion"
            )
            # storing the data
            self.feature_store_file_path = os.path.join(
                self.data_ingestion_dir, "feature_Store", FILE_NAME
            )
            # dividing the data in training data
            self.train_file_path = os.path.join(
                self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME
            )
            # dividing the data in testing data
            self.test_file_path = os.path.join(
                self.data_ingestion_dir, "dataset", TEST_FILE_NAME
            )
            # assigning the test size
            self.test_size = 0.2
        except Exception as e:
            raise InsuranceException(e, sys)

    # convert data into dict
    def to_dict(
        self,
    ) -> dict:
        try:
            return self.__dict__
        except Exception as e:
            raise InsuranceException(e, sys)


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir, "data_validation"
        )
        self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")
        self.missing_threshold: float = 0.2
        self.base_file_path = os.path.join("insurance.csv")
