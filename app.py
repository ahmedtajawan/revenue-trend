import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

# Set the title of the Streamlit app
st.title("📊 Revenue Analysis")

# Sidebar for file upload
st.sidebar.title("Upload Data")

# Load default file if no file is uploaded
default_file = 'ExpandedRevenueData.csv'  # Default file to load
file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Function to display sample file format
def show_sample_format():
    st.info("Sample file format:")
    sample_data = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Revenue Source": ["Source A", "Source B"],
        "Platform/Channel": ["Platform 1", "Platform 2"],
        "Revenue Amount": [1000, 2000],
        "Currency": ["USD", "USD"]
    })
    st.write(sample_data)

# Load data from uploaded file or default file
required_columns = ["Date", "Revenue Source", "Platform/Channel", "Revenue Amount", "Currency"]

try:
    if file is not None:
        # Load the data from the uploaded file
        data = pd.read_csv(file)
        if not all(col in data.columns for col in required_columns):
            st.sidebar.error("The uploaded file does not have the required columns. Please check the format.")
            show_sample_format()  # Display the sample format
            st.stop()
        st.sidebar.markdown("### Uploaded File Format:")
        show_sample_format()  # Display the sample format after upload attempt
    else:
        # Load the default file
        data = pd.read_csv(default_file)
        st.sidebar.markdown("Using default file 'ExpandedRevenueData.csv'.")
except Exception as e:
    st.sidebar.error("The file you uploaded is not in the required format. Please ensure it matches the sample format.")
    show_sample_format()
    st.stop()

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Create a 'Month' column from 'Date'
data['Month'] = data['Date'].dt.strftime('%Y-%m')  # Year-Month format (e.g., '2023-01')

# Set the style for seaborn
sns.set(style="whitegrid")

# Sidebar for navigation with a header and icons
st.sidebar.subheader("Navigate through the app")
page = st.sidebar.radio("Select a Page", ["🏠 Revenue Trends", "📈 Revenue Forecasting"])

# Page 1: Revenue Trends Visualization
if page == "🏠 Revenue Trends":
    st.title("🏠 Revenue Trends Visualization")
    
    # Group by 'Date' and 'Revenue Source'
    df_grouped = data.set_index('Date').groupby(['Revenue Source', pd.Grouper(freq='W')])['Revenue Amount'].sum().reset_index()
    
    # Calculate smoothed revenue
    df_grouped['Smoothed Revenue'] = df_grouped.groupby('Revenue Source')['Revenue Amount'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Plotting
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_grouped, x='Date', y='Smoothed Revenue', hue='Revenue Source', marker=None, linewidth=2)
    plt.title('Smoothed Weekly Revenue Trends by Revenue Source', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Smoothed Revenue Amount', fontsize=12)
    plt.legend(title='Revenue Source', loc='upper left', fontsize=10)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show plot
    st.pyplot(plt)

    # Cross-tabulation
    crosstab = pd.crosstab(data['Revenue Source'], data['Platform/Channel'], values=data['Revenue Amount'], aggfunc='sum')
    plot_data = crosstab.fillna(0)  # Fill NaN with 0 for plotting

    # Plotting stacked bar chart
    st.subheader("Revenue by Source and Platform/Channel")
    st.bar_chart(plot_data)

    # Heatmap
    st.subheader("Heatmap of Revenue by Source and Platform/Channel")
    plt.figure(figsize=(12, 6))
    sns.heatmap(crosstab, cmap='YlGnBu', annot=True, fmt=".0f", linewidths=.5)
    st.pyplot(plt)

    # Monthly Revenue Trends by Platform/Channel
    st.subheader("Monthly Revenue Trends by Platform/Channel")
    revenue_sources = data['Revenue Source'].unique()

    # Loop through each revenue source and create a FacetGrid for monthly data
    for revenue_source in revenue_sources:
        # Filter the data for the current revenue source
        revenue_data = data[data['Revenue Source'] == revenue_source]

        # Create a FacetGrid for the current revenue source across different platforms
        g = sns.FacetGrid(revenue_data, col='Platform/Channel', height=5, sharey=True)
        g.map(sns.lineplot, 'Month', 'Revenue Amount', marker='o')

        # Add titles and labels
        g.add_legend(title='Revenue Source')
        g.set_titles(col_template="{col_name} - " + revenue_source)
        g.set_axis_labels('Month', 'Revenue Amount')

        # Rotate x-ticks for better readability
        for ax in g.axes.flatten():
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Show the plot
        plt.tight_layout()
        st.pyplot(plt)

# Page 2: Revenue Forecasting
elif page == "📈 Revenue Forecasting":
    st.title("📈 Revenue Forecasting")

    forecasts = []
    for revenue_source in data['Revenue Source'].unique():
        for platform in data['Platform/Channel'].unique():
            revenue_data = data[(data['Revenue Source'] == revenue_source) & (data['Platform/Channel'] == platform)]
            df_prophet = revenue_data[['Date', 'Revenue Amount']].rename(columns={'Date': 'ds', 'Revenue Amount': 'y'})
            
            # Check if there are sufficient data points
            if df_prophet['y'].notnull().sum() < 2:
                continue  # Skip without warning if insufficient data
            
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(df_prophet)
            future_dates = model.make_future_dataframe(periods=6, freq='M')
            forecast = model.predict(future_dates)
            forecast['Revenue Source'] = revenue_source
            forecast['Platform/Channel'] = platform
            forecasts.append(forecast[['ds', 'yhat', 'Revenue Source', 'Platform/Channel']])

    forecast_df = pd.concat(forecasts, ignore_index=True)
    forecast_next_6_months = forecast_df[forecast_df['ds'] > data['Date'].max()]

    # Display forecasts
    st.subheader("Forecast for the Next 6 Months")
    st.write(forecast_next_6_months[['ds', 'Revenue Source', 'Platform/Channel', 'yhat']])

    # Plot historical and forecasted data
    st.subheader("Historical and Forecasted Revenue")
    for revenue_source in data['Revenue Source'].unique():
        for platform in data['Platform/Channel'].unique():
            historical_data = data[(data['Revenue Source'] == revenue_source) & (data['Platform/Channel'] == platform)]
            forecast_data = forecast_df[(forecast_df['Revenue Source'] == revenue_source) & (forecast_df['Platform/Channel'] == platform)]
            
            if forecast_data.empty:
                continue
            
            plt.figure(figsize=(10, 6))
            plt.plot(historical_data['Date'], historical_data['Revenue Amount'], label="Historical", color="blue", marker="o", linestyle="None")
            plt.plot(forecast_data['ds'], forecast_data['yhat'], label="Forecast", color="orange", linestyle="--")
            plt.title(f"{revenue_source} - {platform} Revenue Forecast")
            plt.xlabel("Date")
            plt.ylabel("Revenue Amount")
            plt.legend()
            plt.xticks(rotation=45)
            st.pyplot(plt)

    # Monthly Revenue Trends by Platform/Channel (for forecast tab)
    st.subheader("Monthly Revenue Trends by Platform/Channel (Forecasting Tab)")
    for revenue_source in revenue_sources:
        # Filter the data for the current revenue source
        revenue_data = data[data['Revenue Source'] == revenue_source]

        # Create a FacetGrid for the current revenue source across different platforms
        g = sns.FacetGrid(revenue_data, col='Platform/Channel', height=5, sharey=True)
        g.map(sns.lineplot, 'Month', 'Revenue Amount', marker='o')

        # Add titles and labels
        g.add_legend(title='Revenue Source')
        g.set_titles(col_template="{col_name} - " + revenue_source)
        g.set_axis_labels('Month', 'Revenue Amount')

        # Rotate x-ticks for better readability
        for ax in g.axes.flatten():
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # Show the plot
        plt.tight_layout()
        st.pyplot(plt)
else:
    st.warning("Please upload a CSV file to proceed.")
