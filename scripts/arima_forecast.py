import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


class ARIMAForecastModel:
    def __init__(self, p=5, d=1, q=0):
        self.model = None
        self.df_arima = None
        self.p = p
        self.d = d
        self.q = q

    def load_data(self, df, date_col="Data", value_col="Przychody odsetkowe"):

        try:
            df_arima = df[[date_col, value_col]].rename(columns={date_col: "Data", value_col: "Wartość"})
            df_arima["Data"] = pd.to_datetime(df_arima["Data"], errors="coerce")
            df_arima.set_index("Data", inplace=True)
            df_arima.dropna(inplace=True)
            self.df_arima = df_arima

            if len(self.df_arima) < 10:
                raise ValueError("Za mało danych do prognozowania!")

        except Exception as e:
            print(f"Błąd przy wczytywaniu danych do ARIMA: {e}")
            self.df_arima = None

    def train_model(self):

        if self.df_arima is None or len(self.df_arima) < 10:
            raise ValueError("Brak wystarczających danych. Najpierw wywołaj load_data().")

        self.model = ARIMA(self.df_arima, order=(self.p, self.d, self.q)).fit()

    def predict_future(self, periods=12):

        if not self.model:
            raise ValueError("Model nie jest wytrenowany. Wywołaj najpierw train_model().")

        forecast = self.model.get_forecast(steps=periods)
        forecast_df = forecast.summary_frame()
        forecast_df = forecast_df.rename(columns={
            "mean": "Przewidywana wartość",
            "mean_ci_lower": "Dolna granica",
            "mean_ci_upper": "Górna granica"
        })
        return forecast_df

    def plot_forecast(self, x_label="Data", y_label="Wartość"):

        if self.df_arima is None:
            raise ValueError("Brak danych do wizualizacji.")

        plt.figure(figsize=(10, 6))
        plt.plot(self.df_arima, label="Dane historyczne")
        forecast = self.model.get_forecast(steps=12)
        forecast_index = pd.date_range(self.df_arima.index[-1], periods=12, freq="M")
        plt.plot(forecast_index, forecast.predicted_mean, label="Prognoza", color="orange")
        plt.fill_between(forecast_index, forecast.conf_int()["lower Wartość"], forecast.conf_int()["upper Wartość"], color="orange", alpha=0.3)
        plt.title("Prognoza ARIMA", fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()
