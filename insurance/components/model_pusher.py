from insurance.entity import artifact_entity, config_entity
from insurance.exception import InsuranceException
from insurance.logger import logging
import os, sys
import pandas as pd
import numpy as np
from insurance.config import TARGET_COLUMN

from sklearn.metrics import r2_score
from insurance.predictor import ModelResolver
from insurance.utils import *
from insurance.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelPusherArtifact,
)
from insurance.entity.config_entity import ModelPusherConfig


# defining the model pusher class
class ModelPusher:
    # defining the constructor
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_training_artifact: ModelTrainerArtifact,
    ):

        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_training_artifact = model_training_artifact
            # saving model in saved_model_dir in model_registry from model_pusher_config class
            self.model_resolver = ModelResolver(
                model_registry=self.model_pusher_config.saved_model_dir
            )
        except Exception as e:
            raise InsuranceException(e, sys)

    # initiating the model_pusher and saving all data in ModelPusherArtifact
    def initiate_model_pusher(
        self,
    ) -> ModelPusherArtifact:
        try:
            logging.info(f"Loading transformer, model and target encoder")
            # defining the transformer, model and target_encoder_data
            # after comparing, saving all the data to initiate the model

            # taking the transformer data from data_transformation_artifact inside transform_object_path
            transformer = load_object(
                file_path=self.data_transformation_artifact.transform_object_path
            )

            # taking the model data from model_trainer_artifact inside model_path
            model = load_object(file_path=self.model_trainer_artifact.model_path)

            # taking the target_encoder data from data_transformation_artifact inside target_encoder_path
            target_encoder = load_object(
                file_path=self.data_transformation_artifact.target_encoder_path
            )

            # defining the model in saved_object
            logging.info(f"Saving model into model pusher directory")
            save_object(
                file_path=self.model_pusher_config.pusher_transformer_path,
                obj=transformer,
            )
            save_object(
                file_path=self.model_pusher_config.pusher_model_path,
                obj=model,
            )
            save_object(
                file_path=self.model_pusher_config.pusher_target_encoder_path,
                obj=target_encoder,
            )

            # saving the model after comparing the model
            logging.info(f"Saving model in saved model dir")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_encoder_path = (
                self.model_resolver.get_latest_save_target_encoder_path()
            )

            # saving the object defined in the first place
            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)
            save_object(file_path=target_encoder_path, obj=target_encoder)

            # first pushing then saving the model in their respective config class
            model_pusher_artifact = ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                saved_model_dir=self.model_pusher_config.saved_model_dir,
            )
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            # return to model pusher artifact
            return model_pusher_artifact

        except Exception as e:
            raise InsuranceException(e, sys)

        try:
            pass
        except Exception as e:
            raise InsuranceException(e, sys)
