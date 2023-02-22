from insurance.entity import artifact_entity, config_entity
from insurance.logger import logging
from insurance.exception import InsuranceException
import os, sys
import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import ks_2samp
from insurance.config import TARGET_COLUMN
from insurance import utils


class DataValidation:
    def __init__(
        self,
        data_validation_config: config_entity.DataValidationConfig,
        data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
    ):

        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise InsuranceException(e, sys)

    # missing values
    def drop_missing_values_columns(
        self, df: pd.DataFrame, report_key_name: str
    ) -> Optional[pd.DataFrame]:
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum() / df.shape[0]
            logging.info(
                f"selecting column name which contains null above to {threshold}"
            )
            drop_column_names = null_report[null_report > threshold].index
            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name] = list(drop_column_names)

            # creating function for dropping the missing values above than threshold values
            df.drop(list(drop_column_names), axis=1, inplace=True)
            # what if there is no missing values in the list
            # creating function for list having no missing values
            if len(df.columns) == 0:
                return None
            return df

        except Exception as e:
            raise InsuranceException(e, sys)

    # checking original data and training and testing data  have same columns
    def is_required_columns_exists(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str
    ) -> bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    # logging.info(f"Column: [{base} is not available.]")
                    missing_columns.append(base_column)

            if len(missing_columns) > 0:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True

        except Exception as e:
            raise InsuranceException(e, sys)

    #
    def data_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str
    ):
        try:
            drift_report = dict()

            base_columns = base_df.columns

            current_columns = current_df.columns

            # checking whether p value is greater than or less than 0.5
            for base_column in base_columns:
                base_data, current_data = (
                    base_df[base_column],
                    current_df[base_column],
                )

                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    # null hypothesis accepted
                    drift_report[base_column] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution": True,
                    }
                else:
                    drift_report[base_column] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution": False,
                    }

            self.validation_error[report_key_name] = drift_report

        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na": np.NAN}, inplace=True)
            logging.info(f"Replace na value in base df")
            logging.info(f"Drop null values columns from base df")
            base_df = self.drop_missing_values_columns(
                df=base_df, report_key_name="Missing_values_within_base_dataset"
            )
            logging.info(f"Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)

            logging.info(f"Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_values_columns(
                df=train_df, report_key_name="Missing_values_within_train_dataset"
            )
            test_df = self.drop_missing_values_columns(
                df=test_df, report_key_name="Missing_values_within_test_dataset"
            )

            exclude_columns = [TARGET_COLUMN]

            base_df = utils.convert_columns_float(
                df=base_df, exclude_columns=exclude_columns
            )

            train_df = utils.convert_columns_float(
                df=train_df, exclude_columns=exclude_columns
            )
            test_df = utils.convert_columns_float(
                df=test_df, exclude_columns=exclude_columns
            )
            logging.info(f"Is all required columns present in train df")
            train_df_columns_status = self.is_required_columns_exists(
                base_df=base_df,
                current_df=train_df,
                report_key_name="Missing_columns_within_train_dataset",
            )
            logging.info(f"Is all required columns present in test df")
            test_df_columns_status = self.is_required_columns_exists(
                base_df=base_df,
                current_df=test_df,
                report_key_name="Missing_columns_within_test_dataset",
            )

            if train_df_columns_status:
                logging.info(
                    f"As all column are available in train df hence detecting data drift"
                )
                self.data_drift(
                    base_df=base_df,
                    current_df=train_df,
                    report_key_name="Data drift within train dataset",
                )
            if test_df_columns_status:
                logging.info(
                    f"As all column are available in train df hence detecting data drift"
                )
                self.data_drift(
                    base_df=base_df,
                    current_df=test_df,
                    report_key_name="Data drift within train dataset",
                )

            # report generation
            logging.info("Write report in yaml file")
            utils.write_yaml_file(
                file_path=self.data_validation_config.report_file_path,
                data=self.validation_error,
            )

            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise InsuranceException(e, sys)
