import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#toto
class Ticker:
    def __init__(self, ticker):
        self.ticker = ticker            # nom de la valeur boursiere, de la crypto, ...
        self.data = None                # les données récupérer de yfinance, avec les indicateurs tehcnique
        self.normalized = None           # les données normalisées comprises entre [0, 1], sur les data choisies
        self.model = None               # le modèle utilisé pour les prédictions
        self.predictions = None;         # les données prédites avec le modèle
        self.extension = ".csv"
        self.filename = "ticker_" + self.ticker + "_data" + self.extension

    def fetch_data(self, interval="1d", start="2017-01-01", end="2021-01-01"):
        self.data = yf.Ticker(self.ticker).history( interval=interval, start=start, end=end)
        if self.data.empty:
            raise ValueError(f"Aucune donnée disponible pour le ticker '{self.ticker}'")
        print("Données récupérées avec succès")

    def calculate_indicators(self):
        # Clean NaN values
        self.data.dropna(inplace=True)
        self.data = add_all_ta_features(self.data, 
                                        open="Open", high="High", low="Low", close="Close", volume="Volume", 
                                        fillna=True)
        self.data.dropna(inplace=True)
        print("Indicateurs calculés")

    def normalize_data(self, var=['Close']):
        if self.normalized == None :
            self.normalized = pd.DataFrame({})
        self.normalized[var] = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.data[var])
        print("Données normalisées")

    def save_data(self):
        self.data.to_csv(self.filename)
        print(f"Données enregistrées dans {self.filename}")

    def load_data(self):
        self.data = pd.read_csv(self.filename)
        print("données chargées à partir de " + self.filename)

    def load_model_and_predict(self, model, window):
        predictions = model.predict(window)
        self.predictions = predictions
        print("Prédictions réalisées")

    def display(self):
        print("indicators : ")
        if self.data is None :
            print("none")
        else :
            print(self.data.columns)
            print(self.data.head(5))
            print("dimensions : " + str(self.data.shape))

        print("normalized : ")
        if self.normalized is None :
            print("none")
        else:
            print(self.normalized.head(5))
            print("dimensions : " + str(self.normalized.shape))



    def visualize_predictions(self):
        plt.figure(figsize=(14,7))
        plt.plot(self.data['Close'], label='Valeur réelle')
     #   plt.plot(self.predictions, label='Prédictions')
        plt.legend()
        plt.show()
        # Sauvegarder le graphique
        plt.savefig("/home/codespace/nvm/current/mon_graphe.png")
        print("Graphique sauvegardé en tant que 'mon_graphe.png'")


