# Importez les classes définies dans le dépôt
from ticker import Ticker


#ticker = "AAPL"
ticker= Ticker("AAPL")
# Récupération des données historiques
ticker.fetch_data("1d", "2020-01-01", "2022-01-01")
ticker.calculate_indicators()
ticker.display()
ticker.normalize_data(['Close', 'trend_ema_slow'])
ticker.display()
ticker.save_data()
ticker.visualize_predictions()
