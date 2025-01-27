import streamlit as st
from scripts.data_loader import DataLoader
from scripts.forecast import ForecastModel
from scripts.arima_forecast import ARIMAForecastModel  # Import modelu ARIMA
import pandas as pd


def main():
    st.set_page_config(page_title="Prognozowanie finansowe", layout="wide")
    st.title("Prognozowanie cen akcji i wyników finansowych")


    company_map = {
        "PKN Orlen": "pkn",
        "Kęty": "kt",
        "PCC Rokita": "pcc",
        "Bank Handlowy": "bh"
    }


    company = st.sidebar.selectbox("Wybierz spółkę", list(company_map.keys()))
    company_key = company_map[company]

    paths = {
        "daily_data": f"data/{company_key}/{company_key}_d.csv",
        "financial_data": f"data/{company_key}/{company_key}_dane.csv",
        "cash_flow": f"data/{company_key}/{company_key}_pp_wskaźniki.csv",
        "profitability": f"data/{company_key}/{company_key}_rent_wskaźniki.csv",
        "valuation": f"data/{company_key}/{company_key}_wr_wskaźniki.csv"
    }

    # Zakładki
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Ceny akcji",
        "Rentowność",
        "Przepływy pieniężne",
        "Wartość rynkowa",
        "Dane finansowe"
    ])

    # --- Zakładka 1: Ceny akcji ---
    with tab1:
        st.subheader(f"Prognoza cen akcji: {company}")
        daily_data = DataLoader.load_data(paths["daily_data"])

        if daily_data is not None:
            st.write("Dane dzienne załadowane:")
            st.write(daily_data.head())

            daily_model = ForecastModel()
            daily_model.load_data(daily_data, date_col="Data", value_col="Zamkniecie")
            if daily_model.df_prophet is not None:
                daily_model.train_model(changepoint_prior_scale=0.05, seasonality_prior_scale=5)
                daily_forecast = daily_model.predict_future(days_ahead=365 * 3)
                st.write("Prognoza na 3 lata:")
                st.write(daily_forecast)
                st.pyplot(daily_model.plot_forecast(
                    confidence_reduction=0.5, x_label="Data", y_label="Cena zamknięcia"
                ))


                if "Branża" in daily_data.columns:
                    st.line_chart(daily_data.set_index("Data")[["Zamkniecie", "Branża"]])
            else:
                st.error("Nie udało się załadować danych do modelu.")
        else:
            st.error("Nie udało się wczytać danych dziennych.")

    # --- Zakładka 2: Rentowność ---
    with tab2:
        st.subheader(f"Prognoza rentowności spółki i branży: {company}")
        profitability_data = DataLoader.load_data(paths["profitability"])

        if profitability_data is not None:
            profitability_data = DataLoader.split_company_and_industry(profitability_data, source_column="ROE")
            profitability_data["Data-Kwartał"] = profitability_data["Data-Kwartał"].apply(DataLoader.transform_date)

            st.write("Dane rentowności załadowane:")
            st.write(profitability_data[["Data-Kwartał", "wynik_spolki", "wynik_branzy"]])

            profitability_model = ForecastModel()
            profitability_model.load_data(profitability_data, date_col="Data-Kwartał", value_col="wynik_spolki")
            profitability_model.train_model(changepoint_prior_scale=0.05, seasonality_prior_scale=5)
            profitability_forecast = profitability_model.predict_future(days_ahead=365 * 3)

            st.write("Prognoza rentowności spółki:")
            st.write(profitability_forecast)
            st.pyplot(profitability_model.plot_forecast(
                confidence_reduction=0.5, x_label="Data", y_label="Rentowność (%)"
            ))

        else:
            st.error("Nie udało się wczytać danych rentowności.")

    # --- Zakładka 3: Przepływy pieniężne ---
    with tab3:
        st.subheader(f"Przepływy pieniężne: {company}")
        cash_flow_data = DataLoader.load_data(paths["cash_flow"])

        if cash_flow_data is not None:
            st.write("Dane przepływów pieniężnych załadowane:")
            st.write(cash_flow_data.head())

            if len(cash_flow_data) > 10:
                cash_flow_model = ForecastModel()
                cash_flow_model.load_data(cash_flow_data, date_col="Data-Kwartał", value_col="Przepływy pieniężne z działalności operacyjnej")
                if cash_flow_model.df_prophet is not None:
                    cash_flow_model.train_model(changepoint_prior_scale=0.1, seasonality_prior_scale=5)
                    cash_flow_forecast = cash_flow_model.predict_future(days_ahead=365 * 3)
                    st.write("Prognoza przepływów pieniężnych na 3 lata:")
                    st.write(cash_flow_forecast)
                    st.pyplot(cash_flow_model.plot_forecast(confidence_reduction=0.5))
                else:
                    st.error("Nie udało się przygotować danych przepływów pieniężnych do prognozowania.")
            else:
                st.warning("Za mało danych do przeprowadzenia prognozy. Wyświetlam surowe dane.")
                st.line_chart(cash_flow_data.set_index("Data-Kwartał"))
        else:
            st.error("Nie udało się wczytać danych przepływów pieniężnych.")

    # --- Zakładka 4: Wartość rynkowa ---
    with tab4:
        st.subheader(f"Wartość rynkowa: {company}")
        valuation_data = DataLoader.load_data(paths["valuation"])

        if valuation_data is not None:
            valuation_data = DataLoader.split_company_and_industry(valuation_data, source_column="Cena / Zysk")
            st.write("Dane wartości rynkowej załadowane:")
            st.write(valuation_data.head())

            if len(valuation_data) > 10:
                valuation_model = ForecastModel()
                valuation_model.load_data(valuation_data, date_col="Data-Kwartał", value_col="wynik_spolki")
                if valuation_model.df_prophet is not None:
                    valuation_model.train_model(changepoint_prior_scale=0.05, seasonality_prior_scale=5)
                    valuation_forecast = valuation_model.predict_future(days_ahead=365 * 3)

                    st.write("Prognoza wartości rynkowej:")
                    st.write(valuation_forecast)
                    st.pyplot(valuation_model.plot_forecast(confidence_reduction=0.5))
                else:
                    st.error("Nie udało się przygotować danych do prognozowania.")
            else:
                st.warning("Za mało danych do przeprowadzenia prognozy. Wyświetlam surowe dane.")
                st.line_chart(valuation_data.set_index("Data-Kwartał")[["wynik_spolki", "wynik_branzy"]])
        else:
            st.error("Nie udało się wczytać danych wartości rynkowej.")

    with tab5:
        st.subheader(f"Dane finansowe: {company}")
    financial_data = DataLoader.load_data(paths["financial_data"])

    if financial_data is not None:
        st.write("Dane finansowe załadowane:")
        st.write(financial_data.head())

        st.write("Dostępne kolumny w danych finansowych:")
        st.write(financial_data.columns.tolist())

        if "Przychody odsetkowe" in financial_data.columns and len(financial_data) > 10:
            arima_model = ARIMAForecastModel()
            arima_model.load_data(financial_data, date_col="Data", value_col="Przychody odsetkowe")
            arima_model.train_model()
            arima_forecast = arima_model.predict_future(periods=12)

            st.write("Prognoza przychodów odsetkowych (ARIMA):")
            st.write(arima_forecast)
            st.pyplot(arima_model.plot_forecast(x_label="Data", y_label="Przychody odsetkowe"))
        else:
            st.warning("Brak wystarczających danych lub brak kolumny 'Przychody odsetkowe'.")
            if "Data" in financial_data.columns:
                st.line_chart(financial_data.set_index("Data"))
    else:
        st.error("Nie udało się wczytać danych finansowych.")


if __name__ == "__main__":
    main()
