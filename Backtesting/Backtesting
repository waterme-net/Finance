import datetime as dt
import talib
import yfinance as yf
import matplotlib.pyplot

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG

class MyMACDStrat(Strategy):

    def init(self):
        price = self.data.Close
        self.macd = self.I(lambda x: talib.MACD(x)[0], price)
        self.macd_signal = self.I(lambda x: talib.MACD(x)[1], price)


    def next(self):
        if crossover(self.macd, self.macd_signal):
            self.buy()

        elif crossover(self.macd_signal, self.macd):
            self.sell()


start = dt.datetime(2024,1,1)
end = dt.datetime.now()
data = yf.download("AAPL", start=start, end=end)

backtest = Backtest(data, MyMACDStrat, commission=0.002, exclusive_orders=True)

print(backtest.run())

try:
    backtest.plot()
except ValueError as e:
    print(f"Plotting error: {e}. Attempting to use an alternative plotting method.")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.title('AAPL Price with Backtesting Results')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
