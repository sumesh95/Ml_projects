import sys 
import logging

def error_message_details(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    fileName=exc_tb.tb_frame.f_code.co_filename
    error_message="error message in [{0}] and line no is [{1}]".format(fileName,exc_tb.tb_lineno,str(error))
    return error_message


class CustomeException(Exception):
    def __init__(self,error,error_details:sys):
        super().__init__(error)
        self.error_message=error_message_details(error,error_details)

    def __str__(self):
        return self.error_message
    
