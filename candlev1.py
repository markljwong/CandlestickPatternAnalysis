import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import mplfinance as mpf
import pandas as pd
import itertools as it
import sys, getopt
from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
from sklearn import cluster

# Class to hold absolute candlestick data or data relative to previous day
class Candle:
	def __init__(self, _open, _high, _low, _close):
		self.open = _open
		self.high = _high
		self.low = _low
		self.close = _close

	def getOpen(self):
		return self.open

	def getHigh(self):
		return self.high

	def getLow(self):
		return self.low

	def getClose(self):
		return self.close

	def get(self):
		return self.open, self.high, self.low, self.close

# Class that models 3 day relative patterns
class Pattern3:
	def __init__(self, day2, day3):
		self.days = []
		self.days.append(day2)
		self.days.append(day3)

	def get(self):
		return self.days

	def patternMatch(self, day1, day2, day3):
		if day2.getOpen() - day1.getOpen() > 0 and self.days[0].getOpen() == 0:
			return False
		if day2.getHigh() - day1.getHigh() > 0 and self.days[0].getHigh() == 0:
			return False
		if day2.getLow() - day1.getLow() > 0 and self.days[0].getLow() == 0:
			return False
		if day2.getClose() - day1.getClose() > 0 and self.days[0].getClose() == 0:
			return False

		if day3.getOpen() - day2.getOpen() > 0 and self.days[1].getOpen() == 0:
			return False
		if day3.getHigh() - day2.getHigh() > 0 and self.days[1].getHigh() == 0:
			return False
		if day3.getLow() - day2.getLow() > 0 and self.days[1].getLow() == 0:
			return False
		if day3.getClose() - day2.getClose() > 0 and self.days[1].getClose() == 0:
			return False
		return True

# Dictionary that holds all possible 3 day patterns
class Pattern3Dict:
	def __init__(self):
		self.patterns = []
		cands = []

		opts = '01'
		opts = list(it.product(opts, repeat = 4))
		for opt in opts:
			cands.append(Candle(float(opt[0]), float(opt[1]), float(opt[2]), float(opt[3])))
		
		for x in cands:
			for y in cands:
				self.patterns.append(Pattern3(x, y))

	def get(self):
		return self.patterns

def main(argv):
	getNewData = False
	showGraphs = False
	tickers = ['DOW', 'MSFT']

	# Commandline argument parsing
	# ====================================================
	try:
		opts, args = getopt.getopt(argv, 'n:g:t:', ['getNewData', 'showgraphs', 'tickers'])
		for opt, arg in opts:
			if opt in ('-n', '--getNewData'):
				if arg == 't':
					getNewData = True
			if opt in ('-g', '--showgraphs'):
				if arg == 't':
					showGraphs = True
			if opt in ('-t', '--tickers'):
				tickers = [arg]

	except getopt.GetoptError:
		print('dataAcq.py -t <t/f>')


	# Processing for each requested ticker symbol
	# ====================================================
	for ticker in tickers:
		# Retrieve data from API if requested or from saved data
		data = pd.DataFrame(columns = ['Open', 'High', 'Low', 'Close']) 
		if getNewData == True:
			ts = TimeSeries(key="GONWYF9TKW2ITTAT", output_format='pandas')
			data, meta_data = ts.get_daily(ticker, outputsize='full')
			data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
			data.drop(columns=['Volume'])

			clust = cluster.KMeans(n_clusters = 4).fit(data)
			data['Prediction'] = clust.labels_

			compression_opts = dict(method='zip', archive_name=ticker + '_raw.csv')
			data.to_csv(ticker + '_raw.csv', index=False)
		else:
			try:
				data = pd.read_csv(ticker + '_raw.csv')
			except:
				print('ERROR - Missing data for: ' + ticker + ' (Data not yet retrieved?)')
				quit()

		# Process patterns
		patterns = Pattern3Dict()
		patternDat = pd.DataFrame(columns = ['Pattern', '1DayHigh', '6DayHigh'])

		patternCount = 0
		for pattern in patterns.get():
			candles = []
			signals = 0
			OneDayPred = 0
			SixDayPred = 0

			for cursor in range(len(data.index)-6):
				candles.append(Candle(data.Open[cursor], data.High[cursor], data.Low[cursor], data.Close[cursor]))
				if len(candles) == 3:
					test = candles.pop(0)
					if pattern.patternMatch(test, candles[0], candles[1]) == True:
						signals += 1
						if (data.High[cursor+1] - data.Low[cursor+1]) / 2 > (data.High[cursor] - data.Low[cursor]) / 2:
							OneDayPred += 1
						if (data.High[cursor+6] - data.Low[cursor+6]) / 2 > (data.High[cursor] - data.Low[cursor]) / 2:
							SixDayPred += 1 

			newDat = pd.DataFrame([[patternCount, OneDayPred/signals, SixDayPred/signals]],
				columns = ['Pattern', '1DayHigh', '6DayHigh'])
			patternDat = patternDat.append(newDat)
			patternCount += 1

		km = cluster.KMeans(n_clusters = 4, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
		clust = km.fit(patternDat)
		patternDat['Prediction'] = clust.labels_

		patternDat.to_csv(ticker + '_pattern.csv', index=False)

		# If requested, show example graphs of candlesticks
		if showGraphs == True:
			data.index = pd.to_datetime(data.index)
			data = data.iloc[::-1]
			mpf.plot(data, type='candlestick', title=ticker)
			fig = px.scatter(patternDat, x='Pattern', y='6DayHigh', color='Prediction', size='1DayHigh')
			fig.update_layout(title="Clustering of candlesticks pattern types")
			fig.show()

if __name__ == "__main__":
    main(sys.argv[1:])