import sys


class CustomException(Exception):
    def __init__(self, error, error_detail:sys):
       super().__init__(error) 
       self.error_message_detail=self.error_msg_detail(error=error, error_detail=error_detail)


    def error_msg_detail(self, error, error_detail:sys)->str:
        '''
        This function returns the detailed error message
        '''
        _,_,exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        error_message = f'Error occured in python script name [{file_name}], line number [{line_no}] with the following error [{error}]'
        return error_message


    def __str__(self):
        return self.error_message_detail


if __name__=='__main__':

    # Testing Custom Excemption class
    try:
        a = 5
        print(a/0)
    except Exception as e:
        print(CustomException(e, sys))
