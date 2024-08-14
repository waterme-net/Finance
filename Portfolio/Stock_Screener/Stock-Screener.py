import yfinance as yf
import pandas as pd
import datetime as dt
import os

# Create directory to save stock data
if not os.path.exists('stock_data'):
    os.makedirs('stock_data')

# Define S&P 500 ticker
sp500_ticker = '^GSPC'
start = dt.datetime.now() - dt.timedelta(days=365)
end = dt.datetime.now()

# Fetch S&P 500 index data
sp500_df = yf.download(sp500_ticker, start=start, end=end)
sp500_df['Pct Change'] = sp500_df['Adj Close'].pct_change()
sp500_return = (sp500_df['Pct Change'] + 1).cumprod().iloc[-1]

# Replace with actual S&P 500 tickers list
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
return_list = []

# DataFrame to store the final results
final_df = pd.DataFrame(columns=['Ticker', 'Latest_Price', 'Score', 'PE_Ratio', 'PEG_Ratio', 'SMA_150', 'SMA_200', '52_Week_Low', '52_Week_High'])

# Fetch and process stock data
for ticker in tickers:
    try:
        print(f"Processing {ticker}...")
        df = yf.download(ticker, start=start, end=end)
        df.to_csv(f'stock_data/{ticker}.csv')

        df['Pct Change'] = df['Adj Close'].pct_change()
        stock_return = (df['Pct Change'] + 1).cumprod().iloc[-1]
        return_compared = round((stock_return / sp500_return), 2)
        return_list.append(return_compared)

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Create a DataFrame for best performers
best_performers = pd.DataFrame(list(zip(tickers, return_list)), columns=['Ticker', 'Returns_Compared'])
best_performers['Score'] = best_performers['Returns_Compared'].rank(pct=True) * 100

# Filter top performers
best_performers = best_performers[best_performers['Score'] >= best_performers['Score'].quantile(0.7)]

# Process best performers
for ticker in best_performers['Ticker']:
    try:
        df = pd.read_csv(f'stock_data/{ticker}.csv', index_col=0)
        df["SMA_150"] = df['Adj Close'].rolling(window=150).mean()
        df["SMA_200"] = df['Adj Close'].rolling(window=200).mean()
        latest_price = df['Adj Close'].iloc[-1]

        # Fetch fundamental data
        stock_info = yf.Ticker(ticker).info
        pe_ratio = stock_info.get('forwardPE', float('nan'))
        peg_ratio = stock_info.get('pegRatio', float('nan'))
        low_52week = round(df['Low'].min(), 2)
        high_52week = round(df['High'].max(), 2)
        score = round(best_performers[best_performers['Ticker'] == ticker]['Score'].tolist()[0])

        # Define conditions for selection
        condition_1 = latest_price > df['SMA_150'].iloc[-1] > df['SMA_200'].iloc[-1]
        condition_2 = latest_price >= (1.3 * low_52week)
        condition_3 = latest_price > (0.75 * high_52week)
        condition_4 = pe_ratio < 40
        condition_5 = peg_ratio < 2

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            final_df = pd.concat([final_df, pd.DataFrame([{
                'Ticker': ticker, 'Latest_Price': latest_price, 'Score': score,
                'PE_Ratio': pe_ratio, 'PEG_Ratio': peg_ratio,
                'SMA_150': df['SMA_150'].iloc[-1], 'SMA_200': df['SMA_200'].iloc[-1],
                '52_Week_Low': low_52week, '52_Week_High': high_52week
            }])], ignore_index=True)
    except Exception as e:
        print(f"Error processing data for {ticker}: {e}")

final_df = final_df.sort_values(by='Score', ascending=False)
pd.set_option('display.max_columns', 10)
print(f"Final DataFrame:\n{final_df}")
final_df.to_csv('final.csv', index=False)
