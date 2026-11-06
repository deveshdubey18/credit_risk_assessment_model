import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    filename='logs/model.log',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a')


def load_data():
    df = pd.read_csv(r'C:\Devesh\Credit_Risk_Assessment_ML_Model\data\processed\processedData.csv')
    logging.info("Data loaded successfully.")
    return df