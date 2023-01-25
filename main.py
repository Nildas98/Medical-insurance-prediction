from insurance.logger import logging
from insurance.exception import InsuranceException
import os, sys
from insurance.utils import get_collections_as_dataframe


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
        get_collections_as_dataframe(
            database_name="INSURANCE", collection_name="INSURANCE_PROJECT"
        )
    except Exception as e:
        print(e)
