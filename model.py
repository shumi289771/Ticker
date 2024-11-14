
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class Model:
    def __init__(self, ticker_data, scaler = MinMaxScaler(), indicators_to_scale = []):
        self.ticker_data = ticker_data
        self.scaler = scaler 
        self.indicators_to_scale = indicators_to_scale
        self.normalized_data = pd.DataFrame({})  
        self.__normalize_data()

    def __normalize_data(self):
        if self.indicators_to_scale == []:
            self.normalized[self.ticker_data.columns] = self.scaler.fit_transform(self.ticker_data.columns)
        else:
            self.normalized[self.indicators_to_scale] = self.scaler.fit_transform(self.indicators_to_scale)
        
        print("Données normalisées")

    def new_normalize_data(self, indicators_to_scale):
        self.indicators_to_scale = indicators_to_scale
        self.__normalize_data()

    def display(self):
        print("scaler : " + self.scaler )
        print("indocator to scale : " + self.scale)
        print("ticker data : ")
        print(self.ticker_data.head(5))
        print("normelized data")
        print(self.normalized_data.head(5))
           