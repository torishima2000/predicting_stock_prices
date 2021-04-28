import yfinance as yf

ticker = yf.Ticker("7203.T")
hist = ticker.history(period="max")
print(hist)
