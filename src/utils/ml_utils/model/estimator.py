import sys
import numpy as np
from src.logging.logger import logging 
from src.exception.exception import CustomException



class CLTVModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            print(CustomException(e,sys))
    
    def predict(self, x:np.array)->np.array:
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            print(CustomException(e,sys))