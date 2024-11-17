
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Embedding, LSTM, Flatten,Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

class Model:
    #qd le model est élu, il doit être affecté du time_step, scaler, indicators utilisés pendant le rating car ces infos seront utilisées pour calcule les predictions pour le trading, pasles predicion pour le test

    def __init__(self, model_name, optimizer, loss):
        self.model = None
        self.model_name = model_name
        self.scaler = None 
        self.indicators = []
        self.time_step = None              #taille de la fenetre de calcul des données d'entrainement
        self.optimizer = optimizer
        self.loss = loss


    def display(self):
        print("model : ", self.model)
        print("model name : ", self.model_name )
        print("scaler : ", self.scaler )
        print("indicator : ", self.indicators)
        print("time_step : ", self.time_step)
        print("optimizer : ", self.optimizer)
        print("loss : ", self.loss)

    def build(self, X_train, y_train, 
                       epochs, batch_size, X_test, y_test ):
        self.compile(self.X_train)
        self.model.fit(self, X_train, y_train, 
                       epochs, batch_size, 
                       validation_data=(X_test, y_test))

    def predict(self, X_test):
        return self.predict(X_test)
    

class Lstm(Model):

        def __init__(self, optimizer='adam', loss='mean_squared_error'):
            Model.__init__(self, "LSTM", optimizer, loss)

        def compile(self, X_train):
            # Initialiser le modèle
            self.model = Sequential()

            # Ajouter des couches LSTM
            self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2], 1)))
            self.model.add(Dropout(0.2))  # Évite le surapprentissage

            self.model.add(LSTM(units=50, return_sequences=False))
            self.model.add(Dropout(0.2))

            # Ajouter une couche dense (sortie)
            self.model.add(Dense(units=1))  # Prédire une seule valeur (par exemple, prix futur)

            # Compiler le modèle
            self.model.compile(optimizer=self.optimizer, loss=self.loss)


class Rating:
    def __init__(self, model, ticker, time_step=60, scaler = MinMaxScaler(), indicators = ['Close'], epochs=50, batch_size=32):
        self.model = model
        self.ticker = ticker
        self.ticker_data = self.ticker.data
        self.normalized_data = pd.DataFrame({})  
        self.time_step = time_step
        self.scaler = scaler
        self.indicators = indicators
        self.X = []
        self.y = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.predicted_data = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.__normalize_data()
        self.__build_window()
        self.__build_train_test_data()   


    def __build_window(self):
        for i in range(len(self.normalized_data) - self.time_step - 1):
            if self.indicators == []:
                self.X.append(self.normalized_data[i : (i + self.time_step)])
            else:
                self.X.append(self.normalized_data[i : (i + self.time_step)][self.indicators])
            self.y.append(self.normalized_data[i + self.time_step : i + self.time_step + 1]['Close'])

        print("timse step ok")
        
    def __normalize_data(self):
        if self.indicators == []:
            self.normalized_data[self.ticker_data.columns] = self.scaler.fit_transform(self.ticker_data[self.ticker_data.columns])
        else:
            self.normalized_data[self.indicators] = self.scaler.fit_transform(self.ticker_data[self.indicators])

        print("Données normalisées")

    def __build_train_test_data(self):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                        self.y,
                                                                        test_size=0.2,
                                                                        shuffle = False)

            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)
            self.X_test = np.array(self.X_test)
            self.y_test = np.array(self.y_test)

            print("data train")
            

    def display(self):
        print("ticker data : ",self.ticker_data.head(5))
        print("normelized data", self.normalized_data.head(5))
        print("window (", self.time_step, ") : ")
        #for i in range(len(self.X)):
        #    print("Window : ", i)
        #    print("X : ", self.X[i]['Close'], " ====> y : ", self.y[i].values[0])
        print("X_train : ", len(self.X_train), " windows de (", self.time_step, ") éléments")
        print("y_train : ", len(self.y_train))
        print("X_test : ", len(self.X_test), " windows de (", self.time_step, ") éléments")
        print("y_test : ", len(self.y_test))

    def run(self):
        self.model.build(self.X_train, self.y_train, 
                       self.epochs, self.batch_size, 
                       self.X_test, self.y_test)
        
        self.predicted_data = self.model.predict(self.X_test)
        self.predicted_data = self.scaler.inverse_transform(self.predicted_data)  # Revenir à l'échelle originale

        # Comparer avec les vraies valeurs
        real_data = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        # Visualiser

        plt.plot(real_data, color='blue', label='Prix réels')
        plt.plot(self.predicted_data, color='red', label='Prix prédits')
        plt.title('Prédiction des prix')
        plt.xlabel('Temps')
        plt.ylabel('Prix')
        plt.legend()
        plt.show()



