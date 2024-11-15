
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class Model:
    def __init__(self, ticker_data, time_step = 60, scaler = MinMaxScaler(), indicators_to_scale = []):
        self.ticker_data = ticker_data
        self.scaler = scaler 
        self.indicators_to_scale = indicators_to_scale
        self.normalized_data = pd.DataFrame({})  
        self.time_step = time_step              #taille de la fenetre de calcul des données d'entrainement
        self.X = []
        self.y = []
        self.__normalize_data()
        self.__build_window()

    def __build_window(self):
        for i in range(len(self.normalized_data) - self.time_step - 1):
            if self.indicators_to_scale == []:
                self.X.append(self.normalized_data[i : (i + self.time_step)])
            else:
                self.X.append(self.normalized_data[i : (i + self.time_step)][self.indicators_to_scale])
            self.y.append(self.normalized_data[i + self.time_step : i + self.time_step + 1]['Close'])

        
    def __normalize_data(self):
        if self.indicators_to_scale == []:
            self.normalized_data[self.ticker_data.columns] = self.scaler.fit_transform(self.ticker_data[self.ticker_data.columns])
        else:
            self.normalized_data[self.indicators_to_scale] = self.scaler.fit_transform(self.ticker_data[self.indicators_to_scale])

        print("Données normalisées")

    def new_normalize_data(self, indicators_to_scale):
        self.indicators_to_scale = indicators_to_scale
        self.__normalize_data()

    def display(self):
        print("scaler : ", self.scaler )
        print("indicator to scale : ", self.indicators_to_scale)
        print("ticker data : ")
        print(self.ticker_data.head(5))
        print("normelized data")
        print(self.normalized_data.head(5))
        print("window (", self.time_step, ") : ")
        for i in range(len(self.X)):
            print("Window : ", i)
            print("X : ", self.X[i]['Close'], " ====> y : ", self.y[i].values[0])

           