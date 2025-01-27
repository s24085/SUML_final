import pandas as pd


class Indicators:
    @staticmethod
    def calculate_margins(df):

        try:
            df["marża_brutto"] = df["Zysk ze sprzedaży"] / df["Przychody ze sprzedaży"] * 100
            df["marża_operacyjna"] = (
                    df["Zysk operacyjny (EBIT)"] / df["Przychody ze sprzedaży"] * 100
            )
            df["marża_netto"] = df["Zysk netto"] / df["Przychody ze sprzedaży"] * 100
            return df
        except Exception as e:
            print(f"Błąd przy obliczaniu marż: {e}")
            return df

    @staticmethod
    def calculate_quarterly_growth(df):

        try:
            df["przychody_qoq"] = df["Przychody ze sprzedaży"].pct_change() * 100
            df["zysk_netto_qoq"] = df["Zysk netto"].pct_change() * 100
            return df
        except Exception as e:
            print(f"Błąd przy obliczaniu dynamiki kwartalnej: {e}")
            return df
