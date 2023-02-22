import os, sys
from insurance.exception import InsuranceException
from insurance.logger import logging
from datetime import datetime

FILE_NAME = "insurance.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"


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


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir, "data_transformation"
        )
        self.transform_object_path = os.path.join(
            self.data_transformation_dir, "transformer", TRANSFORMER_OBJECT_FILE_NAME
        )
        self.transformed_train_path = os.path.join(
            self.data_transformation_dir,
            "transformed",
            TRAIN_FILE_NAME.replace("csv", "npz"),
        )
        self.transformed_test_path = os.path.join(
            self.data_transformation_dir,
            "transformed",
            TEST_FILE_NAME.replace("csv", "npz"),
        )
        self.target_encoder_path = os.path.join(
            self.data_transformation_dir,
            "target_encoder",
            TARGET_ENCODER_OBJECT_FILE_NAME,
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # creating the model trainer dir in the artifact dir
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir, "model_trainer"
        )
        # defining the model and save as pkl file it in the model trainer dir
        self.model_path = os.path.join(self.model_trainer_dir, "model", MODEL_FILE_NAME)
        # defining the threshold limit for accuracy
        self.expected_accuracy = 0.7
        # accuracy less than 70%, model will not accept old model will continue
        # accuracy more than or equal to 70% will accept the model

        # defining for checking overfitting
        self.overfitting_threshold = 0.3
        # if threshold is more than 0.3 then we will reject
