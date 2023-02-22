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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class ModelTrainer:
    # defining the constructor for calling model_trainer_config class from config_entity
    def __init__(
        self,
        model_trainer_config: config_entity.ModelTrainerConfig,
        data_transformation_artifact: artifact_entity.DataTransformationArtifact,
    ):
        # calling the transformed data for training from artifact_entity

        try:
            # defining the model trainer config
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            # defining the data_transformation_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise InsuranceException(e, sys)

    # defining linear regression model
    def train_model(self, x, y):
        try:
            # defining the linear regression model
            lr = LinearRegression()
            # fitting the model
            lr.fit(x, y)
            return lr
        except Exception as e:
            raise InsuranceException(e, sys)

    # initiating the model, then load the data in model_trainer for training the model
    # for that go to utils file, we will define the model_trainer function

    def initiate_model_trainer(self) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")

            # loading the data load_numpy_array_data from utils file
            # fetching the transformed_trained and test data from data_transformation_artifact
            train_arr = utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_path
            )
            test_arr = utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_path
            )

            logging.info(
                f"Splitting input and target feature from both train and test arr."
            )

            # for x_train and x_test apart from target column we need all the data
            # for y_train and y_test we need all the rows and just target column
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"Train the model")
            model = self.train_model(x=x_train, y=y_train)

            logging.info(f"Calculating f1 train score")
            y_hat_train = model.predict(x_train)

            # as it is a regression problem, R2_square will be used for evaluation
            r2_train_score = r2_score(y_true=y_train, y_pred=y_hat_train)

            logging.info(f"Calculating f1 test score")
            y_hat_test = model.predict(x_test)

            # as it is a regression problem, R2_square will be used for evaluation
            r2_test_score = r2_score(y_true=y_test, y_pred=y_hat_test)
            logging.info(
                f"train score:{r2_train_score} and tests score {r2_test_score}"
            )

            # now we need to check whether it is overfitting or underfitting
            # for this go to config_entity

            logging.info(f"Checking if our model is underfitting or not")
            # if r2_test_score is less than expected_threshold then we will reject
            if r2_test_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"model is not able to give expected accuracy: \
                    {self.model_trainer_config.expected_accuracy}: \
                        model actual score: {r2_test_score}"
                )

            logging.info(f"Checking if our model is overfitting or not")
            diff = abs(r2_train_score - r2_test_score)

            # if the difference greater than mentioned overfitting_threshold
            # then raise exception as train and test score is more than overfitting threshold
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(
                    f"train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}"
                )

            # saving the model, if the result from diff is more than overfitting_threshold
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)
            # it will get saved as object model defined at config_entity file on model_trainer_config

            # preparing location on artifact file for saving model and both r2 train and test score
            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                r2_train_score=r2_train_score,
                r2_test_score=r2_test_score,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

            # now go to main.py file for defining trainer file

        except Exception as e:
            raise InsuranceException(e, sys)
