import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv(r'C:\Devesh\Credit_Risk_Assessment_ML_Model\data\data.csv')
    return df