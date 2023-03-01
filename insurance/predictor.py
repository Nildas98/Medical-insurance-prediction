import os
from typing import Optional
from insurance.logger import logging
from insurance.exception import InsuranceException
from insurance.entity.config_entity import (
    TRANSFORMER_OBJECT_FILE_NAME,
    MODEL_FILE_NAME,
    TARGET_ENCODER_OBJECT_FILE_NAME,
)

# whenever we are running pipeline with new data.
# then predictor file will create a new folder and save the new model
# then we will check accuracy and compare our new model with old model
# if the new model have better accuracy than old model
# then we will accept the new model and deploy it in the pipeline.
# if the accuracy is not better than old model then we will reject

# saved_models format - first integer folder inside that model folder


class ModelResolver:
    def __init__(
        self,
        # defining constructor for creating folder for saving the new model
        model_registry: str = "saved_models",
        # saving transformed data into transformer folder
        transformer_dir_name="transformer",
        # saving encoded target data into target encoder folder
        target_encoder_dir_name="target_encoder",
        # saving the model data into model folder
        model_dir_name="model",
    ):
        # calling and defining all the constructors
        self.model_registry = model_registry
        # if the folder is not defined then create one
        os.makedirs(self.model_registry, exist_ok=True)
        self.transformer_dir_name = transformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name

    # 1
    # defining the function for latest directory
    def get_latest_dir_path(self) -> Optional[str]:
        try:
            logging.info(f"{'>>'*20} Model Predictor {'<<'*20}")
            # defining the model_registry
            dir_name = os.listdir(self.model_registry)
            # if the length of the directory name == 0, then return none
            if len(dir_name) == 0:
                return None
            # saved_models folder name format,
            # first folder,integer represents number of time pipeline runs
            # second folder, directory name
            dir_name = list(map(int, dir_name))
            # latest directory name represented by highest integer number
            latest_dir_name = max(dir_name)
            return os.path.join(self.model_registry, f"{latest_dir_name}")

        except Exception as e:
            raise e

    # 2
    # defining the function for getting latest model every time
    def get_latest_model_path(self):
        try:
            # latest directory
            latest_dir = self.get_latest_dir_path()
            # if latest directory not found
            if latest_dir is None:
                raise Exception(f"Model is not available")

            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise e

    # 3
    # defining the function for getting latest transformer data every time
    def get_latest_transformer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transform data is not available")

            return os.path.join(
                latest_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME
            )
        except Exception as e:
            raise e

    # 4
    # defining the function for getting latest target_encoded data every time
    def get_latest_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Target encoder data is not available")

            return os.path.join(
                latest_dir, self.target_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME
            )
        except Exception as e:
            raise e

    # 5
    # defining the function for saving folder
    def get_latest_save_dir_path(self) -> str:
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registry, f"{0}")

            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry, f"{latest_dir_num+1}")
        except Exception as e:
            raise e

    # 6
    # defining the function for saving model data as model.pkl
    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)

        except Exception as e:
            raise e

    # 7
    # defining the function for saving transform data as transform.pkl
    def get_latest_save_transformer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(
                latest_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME
            )

        except Exception as e:
            raise e

    # 8
    # defining the function for saving target encoded data as target_encoded.pkl
    def get_latest_save_target_encoder_path(self) -> str:
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(
                latest_dir, self.target_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME
            )

        except Exception as e:
            raise e


# now lets go to artifact_entity for defining the artifact
