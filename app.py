import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Indian Stock ARIMA Forecaster (2027-2030)", layout="wide")
st.title("📈 Indian Stock Price Forecasting Dashboard (ARIMA)")
st.write("This application downloads the last 5 years of historical data from Yahoo Finance, forecasts stock prices from **2027 to 2030**, and displays trends, distributions, and exact figures.")

# 1. Indian Stock Selector Configuration
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter NSE Stock Ticker (e.g., RELIANCE, TCS, INFYS, HDFCBANK):", value="RELIANCE").upper().strip()

# Format ticker automatically for Yahoo Finance
if ticker_input and not ticker_input.endswith(".NS"):
    ticker = f"{ticker_input}.NS"
else:
    ticker = ticker_input

# Define Timeline boundaries (Last 5 Years)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

if ticker:
    st.markdown(f"## Fetching data for: **{ticker_input}**")
    
    with st.spinner("Downloading 5 years of historical data from Yahoo Finance..."):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
    if stock_data.empty:
        st.error(f"Could not find data for ticker '{ticker_input}'. Please make sure it is a valid NSE stock ticker symbol.")
    else:
        # Resample to weekly data to reduce computation overhead and stabilize long-term ARIMA variance
        df_close = stock_data['Close'].resample('W-MON').mean().dropna()
        
        st.success("Data successfully downloaded!")
        
        # Display historical data preview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("💡 Historical Trend (Last 5 Years)")
            st.line_chart(df_close)
        with col2:
            st.subheader("📋 Recent Data Snapshot")
            st.dataframe(stock_data['Close'].tail(10), use_container_width=True)
            
        # 2. ARIMA Modeling & Forecasting Logic
        st.markdown("---")
        st.subheader("🤖 ARIMA Forecasting Model Execution")
        
        with st.spinner("Running Auto-ARIMA optimization algorithms to project out to 2030..."):
            try:
                # Find optimal (p,d,q) parameters automatically
                model = auto_arima(df_close, seasonal=False, error_action='ignore', suppress_warnings=True)
                fitted_model = model.fit(df_close)
                
                # Calculate required weekly steps from the last historical data point to December 31, 2030
                last_date = df_close.index[-1]
                target_date = datetime(2030, 12, 31)
                weeks_diff = int((target_date - last_date).days / 7) + 1
                
                # Generate Forecast values and Confidence Intervals
                forecast_values, conf_int = fitted_model.predict(n_periods=weeks_diff, return_conf_int=True)
                
                # Construct forecast timeline index
                forecast_index = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=weeks_diff, freq='W-MON')
                forecast_series = pd.Series(forecast_values, index=forecast_index)
                
                # Filter for requested horizon: 2027 to 2030
                horizon_forecast = forecast_series[(forecast_series.index.year >= 2027) & (forecast_series.index.year <= 2030)]
                
                # Plot 1: Line Graph of Historical vs. Forecasted Data
                st.subheader("📉 Time-Series Projection Plot")
                fig1, ax1 = plt.subplots(figsize=(12, 5))
                ax1.plot(df_close.index, df_close.values, label="Historical Data", color="blue")
                ax1.plot(forecast_series.index, forecast_series.values, label="ARIMA Forecast Track", color="red", linestyle="--")
                ax1.fill_between(forecast_series.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
                ax1.set_title(f"{ticker_input} Price Projection up to 2030 (ARIMA Order: {model.order})", fontsize=14)
                ax1.set_xlabel("Timeline Year")
                ax1.set_ylabel("Stock Price (INR)")
                ax1.legend(loc="upper left")
                ax1.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig1)
                plt.close(fig1)
                
                # Plot 2: Histogram of the forecasted values (2027 - 2030)
                st.markdown("---")
                st.subheader("📊 Distribution of Forecasted Prices (2027 - 2030)")
                
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                sns.histplot(horizon_forecast, kde=True, ax=ax2, color="purple", bins=20)
                ax2.axvline(horizon_forecast.mean(), color='red', linestyle='--', linewidth=2, label=f'Forecast Mean ({horizon_forecast.mean():.2f})')
                ax2.axvline(horizon_forecast.median(), color='green', linestyle='-', linewidth=2, label=f'Forecast Median ({horizon_forecast.median():.2f})')
                ax2.set_title(f"Histogram of Projected Prices for {ticker_input} (2027-2030)")
                ax2.set_xlabel("Forecasted Price Range (INR)")
                ax2.set_ylabel("Frequency")
                ax2.legend()
                st.pyplot(fig2)
                plt.close(fig2)
                
                # 3. Numeric Output Display (Aggregated to monthly averages for cleaner viewing)
                st.markdown("---")
                st.subheader("🎯 Forecast Numbers (2027 - 2030 Monthly Averages)")
                
                # Extract corresponding confidence bounds into a structure we can resample cleanly
                forecast_df = pd.DataFrame({
                    'Forecast': forecast_series,
                    'Lower_Bound': conf_int[:, 0],
                    'Upper_Bound': conf_int[:, 1]
                }, index=forecast_index)
                
                # Filter for years 2027-2030 and resample to monthly mean averages
                monthly_forecast = forecast_df[(forecast_df.index.year >= 2027) & (forecast_df.index.year <= 2030)].resample('ME').mean()
                
                # Format final table display columns
                display_df = pd.DataFrame({
                    'Month': monthly_forecast.index.strftime('%B %Y'),
                    'Forecasted Price (INR)': monthly_forecast['Forecast'].round(2),
                    'Lower Bound Range': monthly_forecast['Lower_Bound'].round(2),
                    'Upper Bound Range': monthly_forecast['Upper_Bound'].round(2)
                }).set_index('Month')
                
                st.dataframe(display_df, use_container_width=True)
                    
            except Exception as ex:
                st.error(f"Mathematical processing failed to converge constraints: {ex}")
