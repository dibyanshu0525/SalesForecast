import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function to load data and perform forecasting
def load_data_and_forecast(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Assuming your CSV file has a Date column, convert it to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Set the Date column as the index
    df.set_index('Date', inplace=True)

    # Assuming your CSV file has numerical Sales data
    sales_series = df['Sales']

    # Perform ARIMA forecasting
    # Adjust parameters (p, d, q) according to your data and requirement
    model = ARIMA(sales_series, order=(90,2,0))  # Example parameters, adjust as needed
    fit_model = model.fit()

    # Forecasting
    forecast_values = fit_model.forecast(steps=10)  # Adjust the number of steps as needed

    return df, forecast_values

# Load data and forecast
data, forecast = load_data_and_forecast('sales_data.csv')

# Plot the current data and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Sales'], label='Actual Sales')
plt.plot(pd.date_range(start=data.index[-1], periods=len(forecast)+1, freq='M')[1:], forecast, label='Forecasted Sales', linestyle='--', color='red')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Trend')
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.show()
