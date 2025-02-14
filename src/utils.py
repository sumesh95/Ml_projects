import sys
import os
import numpy as np
import dill


def save_object(file_path,obj):

    try:
        dir_path=os.makedirs(file_path,exist_ok=True)

        with open (file_path ,"wb") as file:
            dill.dump(obj,file)

    except Exception as e:
        pass 