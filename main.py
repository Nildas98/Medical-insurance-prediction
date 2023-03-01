from insurance.logger import logging
from insurance.exception import InsuranceException
import os, sys
from insurance.utils import get_collections_as_dataframe
from insurance.entity.config_entity import DataIngestionConfig
from insurance.entity import config_entity
from insurance.components.data_ingestion import DataIngestion
from insurance.components.data_validation import DataValidation
from insurance.components.data_transformation import DataTransformation
from insurance.components.model_trainer import ModelTrainer
from insurance.components.model_evaluation import ModelEvaluation
from insurance.components.model_pusher import ModelPusher


# def test_logger_exception():

#     try:
#         logging.info("starting the test logger and exception")
#         result = 3 / 0
#         print(result)
#         logging.info("ending the test logger and exception")
#     except Exception as e:
#         logging.debug(str(e))
#         raise InsuranceException(e, sys)


if __name__ == "__main__":
    try:
        # test_logger_exception()
        # get_collections_as_dataframe(
        #     database_name="INSURANCE", collection_name="INSURANCE_PROJECT"
        # )

        # TRAINING PIPELINE CONFIG
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        # DATA INGESTION
        # first we will call data_ingestion_config file
        data_ingestion_config = config_entity.DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )
        print(data_ingestion_config.to_dict())
        # now defining data_validation class
        # it is coming from data_validation_config
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

        # initiating data_ingestion_artifact
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # DATA VALIDATION
        # first we will call data_validation_config file
        data_validation_config = config_entity.DataValidationConfig(
            training_pipeline_config=training_pipeline_config
        )
        # now defining data_validation class
        # it is coming from data_validation_config, data is from data_ingestion_artifact
        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact,
        )
        # initiating data_validation_artifact
        data_validation_artifact = data_validation.initiate_data_validation()

        # DATA TRANSFORMATION
        # first we will call data_transformation_config file
        data_transformation_config = config_entity.DataTransformationConfig(
            training_pipeline_config=training_pipeline_config
        )
        # now defining data_transformation class
        # it is coming from data_transformation_config, data is from data_ingestion_artifact
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_ingestion_artifact=data_ingestion_artifact,
        )
        # initiating data_transformation_artifact
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )

        # MODEL TRAINER
        # first we will call model_trainer_config file
        model_trainer_config = config_entity.ModelTrainerConfig(
            training_pipeline_config=training_pipeline_config
        )
        # now defining model_trainer class
        # it is coming from model_trainer_config, data is from data_transformation_artifact
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact,
        )
        # initiating model_trainer
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        # MODEL EVALUATION
        # first we will call model_evaluation_config file
        model_evaluation_config = config_entity.ModelEvaluationConfig(
            training_pipeline_config=training_pipeline_config
        )
        # now defining model_trainer class
        # it is coming from model_evaluation_config, data is from data_ingestion_artifact, data_transformation_artifact, model_trainer_artifact
        model_evaluation = ModelEvaluation(
            model_evaluation_config=model_evaluation_config,
            data_ingestion_artifact=data_ingestion_artifact,
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact,
        )
        # initiating model_evaluation
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

        # MODEL PUSHER
        # first we will call model_pusher_config file
        model_pusher_config = config_entity.ModelPusherConfig(
            training_pipeline_config=training_pipeline_config
        )
        # now defining model_pusher class
        # it is coming from model_pusher_config, data is from data_transformation_artifact, model_training_artifact
        model_pusher = ModelPusher(
            model_pusher_config=model_pusher_config,
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact,
        )
        # initiating model_pusher
        model_pusher_artifact = model_pusher.initiate_model_pusher()

    except Exception as e:
        print(e)
