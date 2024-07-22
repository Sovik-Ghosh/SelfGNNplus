import warnings
import os
import numpy as np
import tensorflow as tf
import pickle
import random
from Params import args
from utils.TimeLogger import log
from DataHandler import negSamp, transpose, DataHandler, transToLsts
from recommendation import Recommender
import gc

# Suppress all warnings
warnings.filterwarnings('ignore', category=FutureWarning)
tf.config.experimental_run_functions_eagerly(True)

#tf.config.set_visible_devices([], 'GPU')

def enable_memory_growth():
    # Enable memory growth for all physical GPUs
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Memory growth enabled for GPUs")

if __name__ == '__main__':
    log('Start')
    
    # Enable memory growth
    enable_memory_growth()
    
    # Initialize the data handler and load data
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    
    # Set seeds for reproducibility
    np.random.seed(100)
    random.seed(100)
    tf.random.set_seed(100)
    
    # Initialize Recommender model
    recommender = Recommender(handler)
    
    # Run your model (assuming Recommender has a `run` method)
    recommender.run()