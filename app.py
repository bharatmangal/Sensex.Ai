import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
import pandas as pd
import yfinance as yf
import streamlit as st
import tensorflow as tf
import plotly.express as px
from textblob import TextBlob

# Streamlit UI

st.set_page_config(
    page_title="Sensex.Ai",
    page_icon=":chart_with_upwards_trend:"
)
st.title(':chart_with_upwards_trend: Sensex.Ai')
st.subheader('Stock Analysis and Prediction')
st.markdown('© Bharat & Krish')

# Sidebar
st.sidebar.title(':chart_with_upwards_trend: Sensex.Ai')
st.sidebar.header("User Input")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOGL):", "GOOGL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2022-12-31"))

# Load stock data
df = yf.download(stock_symbol, start=start_date, end=end_date)

if len(df) > 0:
    # Describing Data
    st.subheader('Data Summary')
    st.dataframe(df.describe(), use_container_width=True)

    # Closing Data
    st.subheader('Closing Price vs Time chart')
    fig_close = px.line(df, x=df.index, y='Close', title=f'Closing Price of {stock_symbol}')
    fig_close.update_xaxes(title='Date')
    fig_close.update_yaxes(title='Price')
    st.plotly_chart(fig_close)

    # 100MA
    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig_ma100 = px.line(df, x=df.index, y=ma100, title='Closing Price with 100MA')
    fig_ma100.update_xaxes(title='Date')
    fig_ma100.update_yaxes(title='Price')
    st.plotly_chart(fig_ma100)

    # 200MA
    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma200 = df.Close.rolling(200).mean()
    fig_ma200 = px.line(df, x=df.index, y=[ma100, ma200, df.Close], 
                        labels={'variable': 'Price Type'}, 
                        title='Closing Price with 100MA & 200MA')
    fig_ma200.update_xaxes(title='Date')
    fig_ma200.update_yaxes(title='Price')
    st.plotly_chart(fig_ma200)

    # Load the prediction model
    st.subheader("Stock Prediction")
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

    # Combine code for prediction and graph here
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)

    # Load the model
    model = tf.keras.models.load_model('D:/keras_model.h5', compile=False)

    # Testing Part
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Prediction vs Original Chart (Plotly)
    st.subheader('Prediction vs Original')
    n_points = len(df) - 100  # Number of points to plot
    y_test_to_plot = y_test[:n_points]
    y_predicted_to_plot = y_predicted[:n_points]
    date_values = df.index[-n_points:].to_list()  # Convert DatetimeIndex to a list

    # Ensure both y_test_to_plot and y_predicted_to_plot have the same length
    min_length = min(len(y_test_to_plot), len(y_predicted_to_plot))
    y_test_to_plot = y_test_to_plot[:min_length].reshape(-1)
    y_predicted_to_plot = y_predicted_to_plot[:min_length].reshape(-1)
    date_values = date_values[:min_length]

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({'Date': date_values, 'Original Price': y_test_to_plot, 'Predicted Price': y_predicted_to_plot})

    fig_prediction = px.line(plot_data, x='Date', y=['Original Price', 'Predicted Price'],
                            title='Prediction vs Original')
    fig_prediction.update_xaxes(title='Date')
    fig_prediction.update_yaxes(title='Price')
    st.plotly_chart(fig_prediction)

    # Sentiment Analysis
    st.sidebar.subheader("Sentiment Analysis")
    news_headline = st.sidebar.text_input("Enter News Headline for Sentiment Analysis:")
    
    if news_headline:
        # Perform sentiment analysis
        sentiment = TextBlob(news_headline).sentiment
        st.sidebar.subheader("Sentiment Analysis Result")
        st.sidebar.write(f"Sentiment Polarity: {sentiment.polarity}")
        st.sidebar.write(f"Sentiment Subjectivity: {sentiment.subjectivity}")
    
    # Gainers, Losers, and Top Stocks
    st.sidebar.subheader("Stocks Analysis")
    st.sidebar.write("Explore top stocks, gainers, and losers")

    # Create a dropdown to select the type of stocks to display
    stock_type = st.sidebar.selectbox("Select Stock Type", ["Top Stocks", "Gainers", "Losers"])

    # Define the number of stocks to display
    num_stocks = st.sidebar.number_input("Number of Stocks", min_value=1, max_value=20, value=5)

    # Fetch data based on the selected stock type
    if stock_type == "Top Stocks":
        st.sidebar.write("Top Stocks:")
        top_stocks = yf.download(stock_symbol, period="1d")
        st.sidebar.dataframe(top_stocks.head(num_stocks))
    elif stock_type == "Gainers":
        st.sidebar.write("Top Gainers:")
        gainers = yf.download(stock_symbol, period="1d")
        st.sidebar.dataframe(gainers.head(num_stocks))
    elif stock_type == "Losers":
        st.sidebar.write("Top Losers:")
        losers = yf.download(stock_symbol, period="1d")
        st.sidebar.dataframe(losers.head(num_stocks))


else:
    st.warning("No data available for the selected stock and date range. Please enter a valid stock symbol.")

# Disclaimer and Footer
st.sidebar.subheader("Disclaimer")
st.sidebar.write("This is a trial to analyze and predict stock data.")
st.sidebar.write("Use this information at your own risk.")

st.sidebar.subheader("Footer")
st.sidebar.markdown("© Sensex.Ai")
