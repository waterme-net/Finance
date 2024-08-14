import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf

tickers = ['META', 'AMZN', 'GOOG', 'NVDA']
amounts = [12, 2, 4, 10]
prices = []
total = []

for ticker in tickers:
    try:
        df = yf.download(ticker, start=dt.datetime(2020, 8, 1), end=dt.datetime.now())
        price = df['Close'].iloc[-1]
        prices.append(price)
        total.append(price * amounts[tickers.index(ticker)])
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plotting the total value of each stock as a bar chart
ax1.bar(tickers, total, color='cyan')

ax1.set_facecolor('black')
ax1.figure.set_facecolor('#121212')

ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

ax1.set_title('Aniket Portfolio', color='#EF6C35', fontsize=20)

# Plotting the total value as a pie chart
patches, texts, autotext = ax2.pie(total, labels=tickers, autopct='%1.1f%%', pctdistance=0.8)
[text.set_color('white') for text in texts]

my_circle = plt.Circle((0,0),0.55, color = 'black')
ax2.add_artist(my_circle)

ax2.set_facecolor('black')
ax2.figure.set_facecolor('#121212')

# Adding text annotations
ax2.text(0, 0, 'Portfolio overview', fontsize=14, color='#FFE536', verticalalignment='center', horizontalalignment='center')
ax2.text(0, -0.1, f'Total USD Amount: {sum(total):.2f} $', fontsize=12, color='white', verticalalignment='center', horizontalalignment='center')

counter = 0.15
for ticker in tickers:
    ax2.text(0, -0.1 - counter, f'{ticker}: {total[tickers.index(ticker)]:.2f} $', fontsize=12, color='white', verticalalignment='center', horizontalalignment='center')
    counter += 0.15

plt.show()
