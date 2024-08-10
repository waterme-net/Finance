import talib
import datetime as dt
import pandas_datareader as web

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG


class MyMac(Strategy):
    def init(self):
        price = self.data.Close
        self.macd = self.I(lambda x: talib.MACD(x)[0], price)
        self.macd_signal = self.I(lambda x: talib.MACD(x)[1], price)

    def next(self):
        if crossover(self.macd, self.macd_signal):
            self.buy()

        elif crossover(self.macd_signal, self.macd):
            self.sell()



start = dt.datetime(2023,1,1)
end = dt.datetime(2024,1,1)
data = web.DataReader("TSLA","yahoo", start, end)

backtest = Backtest(data, MyMac, commission=.002, exclusive_orders=True)

print(backtest.run())

backtest.plot()
