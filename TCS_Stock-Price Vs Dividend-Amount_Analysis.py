#!/usr/bin/env python
# coding: utf-8

# # TCS Stock Price Vs Dividend Amount Analysis

# # Level - 1 Stock Price Prediction

# Importing Libraries of Pandas, Numpy and Matplotlib

# In[15]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# Collecting TCS stock price data (20 Years) from 25/08/2004 to 30/08/2024 collected from Yahoo Finance and store it into csv

# In[16]:


ticker = 'TCS.BO'
start_date = '2004-08-25'
end_date = '2024-08-30'

stock_data = yf.download(ticker, start=start_date, end=end_date)
stock_data.to_csv('TCS.BO_stock_data.csv')


# In[17]:


csv_file_price = 'TCS.BO_stock_data.csv'
data_price = pd.read_csv(csv_file_price)
print(data_price.head(10))


# Print only the prices of Date with Open, High, Low, and Close Only and It's Average

# In[18]:


ohlc = price[['Date','Open','High','Low','Close']].copy()
col1 = ohlc['Open']
col2 = ohlc['High']
col3 = ohlc['Low']
col4 = ohlc['Close']
ohlc.loc[:,'Avg_OHLC'] = (col2+col1+col3+col4) / 4
merged = pd.merge(ohlc,ohlc)
print(merged.head(10))


# Print Yearly Average of OHLC

# In[19]:


avg_price = pd.DataFrame(ohlc)
avg_price['Date'] = pd.to_datetime(avg_price['Date'])
avg_price['Year'] = avg_price['Date'].dt.year

yearly_price = avg_price.groupby('Year')['Avg_OHLC'].mean().reset_index()
yearly_price.columns = ['Year','Price']
row = len(yearly_price)
print(f"Total No. of Row of Data: {row}")
print(yearly_price)


# Plot the graph of Year Vs Price

# In[21]:


plt.figure(figsize=(10, 6))
plt.plot(yearly_price['Year'], yearly_price['Price'], color = 'green', marker = '*')
plt.title('Yearly Average Price (INR ₹)')
plt.xlabel('Year')
plt.ylabel('Price INR ₹')
plt.xticks(yearly_price.Year, rotation=90)
plt.show()


# # Machine Learning - Linear regression Model to Predict Future Price

# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming 'yearly_price' is the data with 'Year' and 'Price' columns
df = pd.DataFrame(yearly_price)

X = df[['Year']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Future years to predict
future_years = [2025, 2026, 2027, 2028, 2029]
future_year_price = pd.DataFrame({'Year': future_years})
predictions_price = model.predict(future_year_price)

# Display predictions and errors
print("Future Price Prediction [2025 to 2029] INR :", predictions_price)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# User input for a specific year prediction
new_input = int(input("Enter a future year to invest in TCS: "))
new_future_year_price = pd.DataFrame({'Year': [new_input]})
new_predictions_price = model.predict(new_future_year_price)
print(f"Future Price Prediction for the Year {new_input}: ₹", round(new_predictions_price[0], 2))

# Investment calculation
invest = int(input("Enter an amount to invest in that future year: ₹"))
stocks = invest / new_predictions_price.item()
print(f"You will buy {stocks:.2f} stocks in the year of {new_input}")

# Plot the results
plt.figure(figsize=(10, 6))
X_range = pd.DataFrame({'Year': np.linspace(X['Year'].min(), X['Year'].max(), 100)})
y_range = model.predict(X_range)
plt.plot(X, y, color='blue', label="Actual")
plt.plot(X_range, y_range, color='green', label="Predicted")
plt.plot(future_year_price['Year'], predictions_price, color='red', label="Future", marker="*")
plt.plot(new_future_year_price['Year'], new_predictions_price, color="purple", marker="*", label="User")
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Price INR ₹")
plt.title("Actual Vs Predicted Vs Future - Yearly Price INR ₹")
plt.legend()
plt.show()


# # Level - 2 Dividend Amount Prediction

# Collect TCS Dividend Amount data (20 Years) from 01/10/2004 to 30/08/2024 from Yahoo Finance and store it into csv

# In[27]:


ticker = 'TCS.BO'
start_date = '2004-10-01'
end_date = '2024-08-30'

stock = yf.Ticker(ticker)
dividend_data = stock.dividends.loc[start_date:end_date]
dividend_data.to_csv('TCS.BO_dividend_data.csv')


# In[28]:


csv_file_divi = 'TCS.BO_dividend_data.csv'
data_divi = pd.read_csv(csv_file_divi)
print(data_divi.head(10))


# In[29]:


add = pd.DataFrame(data_divi)
add['Date'] = pd.to_datetime(add['Date'])
add['Year'] = add['Date'].dt.year

annual_dividends = add.groupby('Year')['Dividends'].sum().reset_index()
annual_dividends.columns = ['Year','Amount']
row = len(annual_dividends)
print(f"Total No. of Row of Data: {row}")
print(annual_dividends)


# In[32]:


plt.figure(figsize=(10, 6))
plt.plot(annual_dividends['Year'], annual_dividends['Amount'], color='blue', marker = '*')
plt.grid(True)
plt.title('Annual Dividends INR ₹')
plt.xlabel('Year')
plt.ylabel('Amount INR ₹')
plt.xticks(annual_dividends.Year, rotation=90)
plt.show()


# # Machine Learning - Linear regression Model to Predict Dividend Amount

# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# DataFrame with Year and Amount
df = pd.DataFrame(annual_dividends)

X = df[['Year']]
y = df['Amount']

# Splitting the data into training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the test set results
y_pred = model.predict(X_test)

# Future predictions
future_year_div = pd.DataFrame({'Year': [2024, 2025, 2026, 2027, 2028]})
predictions_div = model.predict(future_year_div)
print("Future Dividend Amount [2024 - 2028]: ", predictions_div)

# Calculating errors
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Predicting for a user-provided year
year = int(input("Enter a Future Year to Find Dividend Amount: "))
year_div = pd.DataFrame({'Year': [year]})
new_div = model.predict(year_div)
print("The Dividend Amount for One Stock is : ₹", new_div.item())

# Calculating total dividend for the provided number of stocks
stocks = float(input("Enter the No. of Stocks to Calculate Dividend: "))
amt = new_div.item() * stocks
print(f"The Total Amount of Dividend is ₹ {amt} for {stocks} stocks")

# Plotting the results
plt.figure(figsize=(10, 6))
X_range = pd.DataFrame({'Year': np.linspace(X['Year'].min(), X['Year'].max(), 100)})
y_range = model.predict(X_range)
plt.plot(X, y, color='blue', label="Actual")
plt.plot(X_range, y_range, color='green', label="Predicted")
plt.plot(future_year_div['Year'], predictions_div, color='red', label="Future", marker = "*")
plt.plot(year_div['Year'], new_div, color="purple", marker="*", label="User")
plt.grid(True)
plt.xlabel("Year")
plt.ylabel("Amount ₹")
plt.title("Actual Vs Predicted Vs Future - Dividend Amount INR ₹")
plt.legend()
plt.show()


# In[ ]:




