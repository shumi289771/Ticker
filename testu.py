# Importez les classes définies dans le dépôt
from ticker import Ticker


#ticker = "AAPL"
ticker= Ticker("AAPL")
# Récupération des données historiques
ticker.fetch_data("1d", "2020-01-01", "2022-01-01")
print(type(ticker.data))
ticker.calculate_indicators()
ticker.display_data()