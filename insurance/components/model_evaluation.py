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
from insurance.predictor import ModelResolver
from insurance.utils import *

# here we are evaluating new data with old data
# for that, on new data we need to perform all the steps we performed in old data
# data_ingestion, data_transformation, model_training etc


# defining the model evaluation class
class ModelEvaluation:
    # defining the constructor for calling all the steps
    def __init__(
        self,
        model_evaluation_config: config_entity.ModelEvaluationConfig,
        data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
        data_transformation_artifact: artifact_entity.DataTransformationArtifact,
        model_trainer_artifact: artifact_entity.ModelTrainerArtifact,
    ):

        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise InsuranceException(e, sys)

    # defining the function for initiating model evaluation process
    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info(
                "if saved model folder has model the we will compare "
                "which model is best trained or the model from saved model folder"
            )
            # calling the latest directory from model resolver
            latest_dir_path = self.model_resolver.get_latest_dir_path()

            # here we are defining the function to check whether any folder is being created or not
            # why None? if the accuracy of new data is not greater than old data
            # so no folder will be created in this situation and we are continuing with old data
            if latest_dir_path == None:
                # here it is referring to artifact_entity class for model_evaluation
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(
                    # here the old model is accepted i.e. True
                    # accuracy is not improved in new model i.e False
                    is_model_accepted=True,
                    improved_accuracy=None,
                )
                logging.info(f"Model Evaluation Artifact {model_evaluation_artifact}")

                return model_evaluation_artifact

            # we are going to compare old model with new model, for that we need location of the data
            # for comparison we need old transform data, old model data and old encoder data

            # now we will find out location of old model

            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            # defining the object of old model
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            # defining the object of new model
            current_transformer = load_object(
                file_path=self.data_transformation_artifact.transform_object_path
            )
            current_model = load_object(
                file_path=self.model_trainer_artifact.model_path
            )
            current_target_encoder = load_object(
                file_path=self.data_transformation_artifact.target_encoder_path
            )

            # for testing we are saving it in test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # for getting dependent and independent data we have to remove the target column
            target_df = test_df[TARGET_COLUMN]

            # defining the target data as y_true
            y_true_old = target_df

            # In test data we have to handle categorical data
            # by first transforming and then encoding
            input_old_features_name = list(transformer.feature_names_in)
            # running the loop
            for i in input_old_features_name:
                # if datatype of test_data is object then encode by fit_transform the test data
                if test_df.dtypes == "object":
                    test_df[i] = target_encoder.fit_transform(test_df[i])

            # transforming the input feature from test data on input_features_name
            old_input_arr = transformer.transform(test_df[input_old_features_name])

            # predicting the model using old_input_arr
            y_pred_old = model.predict(old_input_arr)

            # comparing old model score with the help of R2 square
            old_model_score = r2_score(y_true=y_true_old, y_pred=y_pred_old)

            # checking accuracy of current model
            input_new_features_name = list(current_transformer.feature_names_in)

            # transforming the new input feature
            new_input_arr = current_transformer.transform(
                test_df[input_new_features_name]
            )

            # predicting the model using new_input_arr
            y_pred_new = current_model.predict(new_input_arr)

            # defining the target column
            y_true_new = target_df

            # comparing the old model score with the help of R2 square
            new_model_score = r2_score(y_true=y_true_new, y_pred=y_pred_new)

            # final comparison between old model score with new model score
            if new_model_score <= old_model_score:
                logging.info(f"New trained model is not better than old model")
                raise Exception("New model is not better than old model")

            model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=new_model_score - old_model_score,
            )

            return model_evaluation_artifact

        except Exception as e:
            raise InsuranceException(e, sys)
