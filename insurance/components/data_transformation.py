from insurance.entity import artifact_entity, config_entity
from insurance.exception import InsuranceException
from insurance.logger import logging
import os, sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from insurance.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder
from insurance import utils


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: config_entity.DataTransformationConfig,
        data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
    ):

        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise InsuranceException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:

        try:
            logging.info(f"missing values impute, for this we are using simple imputer")
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)

            logging.info(f"outlier detection, for this we are using robust scaler")
            robust_scaler = RobustScaler()

            logging.info(
                f"defining the pipeline using simple imputer and robust scaler"
            )
            pipeline = Pipeline(
                steps=[("imputer", simple_imputer), ("scaler", robust_scaler)]
            )
            return pipeline

        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_transformation(
        self,
    ) -> artifact_entity.DataTransformationArtifact:
        try:
            logging.info(f"reading training and testing data from artifact folder")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(
                f"dropping target column from train and test data from config folder"
            )
            logging.info(f"defining input feature")
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            logging.info(f"defining target feature")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(f"data transformation, for this we are using label encoder")
            label_encoder = LabelEncoder()

            logging.info(f"transforming target data")
            target_feature_train_arr = target_feature_train_df.squeeze()
            target_feature_test_arr = target_feature_test_df.squeeze()

            logging.info(f"converting categorical data into numerical data")
            for col in input_feature_train_df.columns:
                if input_feature_test_df[col].dtypes == "O":
                    input_feature_train_df[col] = label_encoder.fit_transform(
                        input_feature_train_df[col]
                    )
                    input_feature_test_df[col] = label_encoder.fit_transform(
                        input_feature_test_df[col]
                    )
                else:
                    input_feature_train_df[col] = input_feature_train_df[col]
                    input_feature_test_df[col] = input_feature_test_df[col]

            # imbalance data handling

            logging.info(f"calling original data for fitting the pipeline")
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            # transforming input data
            input_feature_train_arr = transformation_pipeline.transform(
                input_feature_train_df
            )
            input_feature_test_arr = transformation_pipeline.transform(
                input_feature_test_df
            )

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info(f"saving the transform_train_path")
            utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_path,
                array=train_arr,
            )

            logging.info(f"saving the transform_test_path")
            utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_path,
                array=test_arr,
            )

            logging.info(f"saving the transform_object_path")
            utils.save_object(
                file_path=self.data_transformation_config.transform_object_path,
                obj=transformation_pipeline,
            )

            logging.info(f"saving the target_encoder_path")
            utils.save_object(
                file_path=self.data_transformation_config.target_encoder_path,
                obj=label_encoder,
            )

            logging.info(f"Data transformation artifact")
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path,
            )

            return data_transformation_artifact

        except Exception as e:
            raise InsuranceException(e, sys)
