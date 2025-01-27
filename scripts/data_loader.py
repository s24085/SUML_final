import pandas as pd
import re


class DataLoader:
    @staticmethod
    def transform_date(date_str):

        try:
            match = re.match(r"(\d{4})/Q(\d)", date_str)
            if match:
                year = int(match.group(1))
                quarter = int(match.group(2))
                month = (quarter - 1) * 3 + 1
                return pd.Timestamp(year, month, 1)
            return pd.to_datetime(date_str, errors="coerce")
        except Exception as e:
            print(f"Błąd przy przekształcaniu daty: {e}")
            return None

    @staticmethod
    def load_data(filepath):

        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Błąd przy wczytywaniu danych z pliku {filepath}: {e}")
            return None

    @staticmethod
    def split_company_and_industry(df, source_column):

        try:
            df["wynik_spolki"] = (
                df[source_column]
                .str.extract(r"^([\d.,-]+)%")
                .replace(",", ".", regex=True)
                .astype(float)
            )
            df["wynik_branzy"] = (
                df[source_column]
                .str.extract(r"branża ([\d.,-]+)%")
                .replace(",", ".", regex=True)
                .astype(float)
            )
            return df
        except Exception as e:
            print(f"Błąd przy podziale danych spółki i branży: {e}")
            return df