
import matplotlib.pyplot as overfitting
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
class Rating:
    def __init__(self, model, ticker, time_step=60, scaler = MinMaxScaler(), indicators = ['Close'], epochs=50, batch_size=32):
        self.model = model
        self.ticker = ticker
        self.ticker_data = self.ticker.data
        self.normalized_data = pd.DataFrame({})  
        self.time_step = time_step
        self.scaler = scaler
        self.close_scaler = scaler
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
        self.train_loss = None
        self.val_loss = None
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

        print("fenetre construite")
        
    def __normalize_data(self):
        if self.indicators == []:
            self.normalized_data[self.ticker_data.columns] = self.scaler.fit_transform(self.ticker_data[self.ticker_data.columns])
        else:
            self.normalized_data[self.indicators] = self.scaler.fit_transform(self.ticker_data[self.indicators])

        self.close_scaler.fit(np.array(self.ticker_data['Close']).reshape(-1,1)) # permet dinverser la normalisation pour afficher les données en clair

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

            print("données séparées")
            

    def display_data(self):
        print("ticker data : ",self.ticker_data.head(5))
        print("normelized data", self.normalized_data.head(5))
        print("window (", self.time_step, ") : ")
        #for i in range(len(self.X)):
        #    print("Window : ", i)
        #    print("X : ", self.X[i]['Close'], " ====> y : ", self.y[i].values[0])

        print("X_train : ", self.X_train.shape[0], " windows de taille : ", self.X_train.shape[1], " x ", self.X_train.shape[2])
        print("y_train : ", self.y_train.shape)
        print("X_test : ", self.X_test.shape[0], " windows de taille : ", self.X_test.shape[1], " x ", self.X_test.shape[2])
        print("y_test : ", self.y_test.shape)

    def run(self):
        self.train_loss, self.val_loss = self.model.build(self.X_train, self.y_train, 
                                                            self.epochs, self.batch_size, 
                                                            self.X_test, self.y_test)
        
        self.predicted_data = self.model.predict(self.X_test)

    def display(self):
        print("scaler       : ",self.scaler)
        print("indicators   : ",self.indicators)
        print("epochs       : ", self.epochs)
        print("batch size   : ", self.batch_size)
        print("time_step    : ", self.time_step)

        predicted_data = self.close_scaler.inverse_transform(self.predicted_data)  # Revenir à l'échelle originale

        # Comparer avec les vraies valeurs
        real_data = self.close_scaler.inverse_transform(self.y_test.reshape(-1, 1))

        # Visualiser
        plt.plot(real_data, color='blue', label='Prix réels')
        plt.plot(predicted_data, color='red', label='Prix prédits')
        plt.title('Prédiction des prix')
        plt.xlabel('Temps')
        plt.ylabel('Prix')
        plt.legend()
        plt.show()

        overfitting.plot(self.train_loss, color='yellow', label='train_loss')
        overfitting.plot(self.val_loss, color='orange', label='val_loss')
        overfitting.title('Overfitting')
        overfitting.xlabel('Temps')
        overfitting.ylabel('loss')
        overfitting.legend()
        overfitting.show()





