from insurance.logger import logging
from insurance.exception import InsuranceException
import os, sys
from insurance.utils import get_collections_as_dataframe
from insurance.entity.config_entity import DataIngestionConfig
from insurance.entity import config_entity


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
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )
        print(data_ingestion_config.to_dict())
    except Exception as e:
        print(e)
