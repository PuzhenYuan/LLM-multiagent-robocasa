print(__file__)
import robocasa
print(robocasa.__file__)
import os
BASE_PATH = os.path.abspath(robocasa.__file__ + '/../../../') 
print(BASE_PATH)
import sys
sys.path.append(BASE_PATH)
import llmagent