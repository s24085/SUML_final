import matplotlib.pyplot as plt
import mplfinance as mpf


class Visualizations:
    @staticmethod
    def plot_candlestick(daily_data):

        try:
            daily_data.index = pd.to_datetime(daily_data["Data"])
            mpf.plot(daily_data, type="candle", style="charles", volume=True, title="Ceny akcji")
        except Exception as e:
            print(f"Błąd przy tworzeniu wykresu świecowego: {e}")

    @staticmethod
    def plot_comparative(data, columns, title):

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in columns:
                ax.plot(data["ds"], data[col], label=col)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Data", fontsize=12)
            ax.set_ylabel("Wartość", fontsize=12)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Błąd przy tworzeniu wykresu porównawczego: {e}")
