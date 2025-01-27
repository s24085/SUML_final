import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


class ForecastModel:
    def __init__(self):
        self.model = None
        self.df_prophet = None
        self.forecast = None

    def load_data(self, df, date_col="Data", value_col="Zamkniecie"):

        try:
            df_prophet = df.rename(columns={date_col: "ds", value_col: "y"})
            df_prophet["y"] = (
                df_prophet["y"]
                .astype(str)
                .str.replace(",", ".")
                .str.extract(r"([\d.]+)")
                .astype(float)
            )
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
            df_prophet.dropna(subset=["ds", "y"], inplace=True)
            self.df_prophet = df_prophet

            if len(self.df_prophet) < 10:
                raise ValueError("Za mało danych do prognozowania!")

        except Exception as e:
            print(f"Błąd przy wczytywaniu danych: {e}")
            self.df_prophet = None

    def train_model(self, changepoint_prior_scale=0.1, seasonality_prior_scale=10):

        if self.df_prophet is None or len(self.df_prophet) < 10:
            raise ValueError("Brak wystarczających danych. Najpierw wywołaj load_data().")

        self.model = Prophet(
            yearly_seasonality=True,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        self.model.add_country_holidays(country_name="PL")
        self.model.fit(self.df_prophet)

    def predict_future(self, days_ahead=365):

        if not self.model:
            raise ValueError("Model nie jest wytrenowany. Wywołaj najpierw train_model().")

        future = self.model.make_future_dataframe(periods=days_ahead)
        self.forecast = self.model.predict(future)

        # Zmieniamy nazwy kolumn na bardziej czytelne
        forecast = self.forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={
                "ds": "Data",
                "yhat": "Przewidywana wartość",
                "yhat_lower": "Dolna granica",
                "yhat_upper": "Górna granica"
            }
        )
        return forecast

    def plot_forecast(self, confidence_reduction=1.0, x_label="Data", y_label="Wartość"):

        if self.forecast is None:
            raise ValueError("Brak prognozy. Najpierw wywołaj predict_future().")

        fig, ax = plt.subplots(figsize=(10, 6))
        self.model.plot(self.forecast, ax=ax)

        reduced_upper = self.forecast["yhat"] + confidence_reduction * (self.forecast["yhat_upper"] - self.forecast["yhat"])
        reduced_lower = self.forecast["yhat"] - confidence_reduction * (self.forecast["yhat"] - self.forecast["yhat_lower"])

        ax.fill_between(self.forecast["ds"], reduced_lower, reduced_upper, color="blue", alpha=0.2, label="Niepewność prognozy")
        ax.set_title("Prognoza z Prophet z niepewnością", fontsize=16)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        plt.legend()
        plt.tight_layout()
        return fig
