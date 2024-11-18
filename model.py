
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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

    def build(self, X_train, y_train, epochs, batch_size, X_test, y_test ):
        self.compile(X_train)
        history = self.model.fit(X_train, y_train, 
                                epochs, batch_size, 
                                validation_data=(X_test, y_test)).history
        
        return history['loss'], history['val_loss']
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    

class Lstm(Model):

        def __init__(self, optimizer='adam', loss='mean_squared_error'):
            Model.__init__(self, "LSTM", optimizer, loss)

        def compile(self, X_train):
            # Initialiser le modèle
            self.model = Sequential()

            # Ajouter des couches LSTM
            self.model.add(LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))#X_train.shape[2]
            self.model.add(Dropout(0.2))  # Évite le surapprentissage

            self.model.add(LSTM(units=50, return_sequences=False))
            self.model.add(Dropout(0.2))

            # Ajouter une couche dense (sortie)
            self.model.add(Dense(units=1))  # Prédire une seule valeur (par exemple, prix futur)

            # Compiler le modèle
            self.model.compile(optimizer=self.optimizer, loss=self.loss)

            self.model.summary()

