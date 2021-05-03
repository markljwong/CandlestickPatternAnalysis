import plotly.express as px
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import logging as log
import numpy as np
import sys, getopt, time, config
from pprint import pprint
from alpha_vantage.timeseries import TimeSeries
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

class Candle:
	def __init__(self, _Trend, _Open, _High, _Low, _Close):
		self.data = {
			'Trend': _Trend,
			'Open': _Open,
			'High': _High,
			'Low': _Low,
			'Close': _Close
		}

	def get(self, key):
		return self.data[key]

	def getBody(self):
		return abs(self.data['Close'] - self.data['Open'])

	def getHighWick(self):
		return min(self.data['High'] - self.data['Close'], self.data['High'] - self.data['Open'])

	def getLowWick(self):
		return min(self.data['Open'] - self.data['Low'], self.data['Close'] - self.data['Low'])

def newDataFrame(patternDays):
	df = pd.DataFrame()
	df['Trend'] = None
	df['Rise_0'] = None
	df['Body_i_0'] = None
	df['h_Wick_i_0'] = None
	df['l_Wick_i_0'] = None

	for x in range(1, patternDays):
		df['Rise_' + str(x)] = None
		df['Open_' + str(x)] = None
		df['High_' + str(x)] = None
		df['Low_' + str(x)] = None
		df['Close_' + str(x)] = None

		df['Body_' + str(x)] = None
		df['h_Wick_' + str(x)] = None
		df['l_Wick_' + str(x)] = None

		df['h_Wick_i_' + str(x)] = None
		df['l_Wick_i_' + str(x)] = None

	df['Label'] = None
	df['LabelConf'] = None

	return df

def newData(ticker):
	ts = TimeSeries(key=api_key, output_format='pandas')
	data, meta_data = ts.get_daily(ticker, outputsize='full')
	data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
	data.drop(columns=['Volume'])
	data = data.iloc[::-1]

	compression_opts = dict(method='zip', archive_name=ticker + '_raw.csv')
	data.to_csv(ticker + '_raw.csv', index=False)

	return data

def acqPatterns(data, patternDays, trainPercentage):
	procDat = newDataFrame(patternDays)

	candles = []
	for cursor in range(6, len(data.index) - 6):
		# Store new candle
		trend = int(data.Close[cursor] >= data.Open[cursor - 6])
		candles.append(Candle(trend, data.Open[cursor], data.High[cursor], data.Low[cursor], data.Close[cursor]))

		# If we have enough candles in list to form a pattern, process all of them
		newDat = pd.DataFrame()
		if len(candles) == patternDays:
			# Process first day
			newDat['Trend'] = [candles[0].get('Trend')]
			newDat['Rise_0'] = [int(candles[0].get('Close') >= candles[0].get('Open'))]
			newDat['Body_i_0'] = [candles[0].getBody()]
			newDat['h_Wick_i_0'] = [candles[0].getHighWick()]
			newDat['l_Wick_i_0'] = [candles[0].getLowWick()]

			# Process days remaining in pattern
			for x in range(1, patternDays):
				# Get interday relative prices
				newDat['Rise_' + str(x)] = [int(candles[x].get('Close') >= candles[x].get('Open'))]
				newDat['Open_' + str(x)] = [candles[x].get('Open') / candles[x-1].get('Open')]
				newDat['High_' + str(x)] = [candles[x].get('High') / candles[x-1].get('High')]
				newDat['Low_' + str(x)] = [candles[x].get('Low') / candles[x-1].get('Low')]
				newDat['Close_' + str(x)] = [candles[x].get('Close') / candles[x-1].get('Close')]

				# If and else to handle divide by 0 cases
				# Set to arbitrary 5.0 
				# NOTE: may need tuning of data if improperly scaling

				# Get interday relative sizes
				if candles[x-1].getBody() == 0:
					newDat['Body_' + str(x)] = [5.0]
				else:
					newDat['Body_' + str(x)] = [candles[x].getBody() / candles[x-1].getBody()]
				if candles[x-1].getHighWick() == 0:
					newDat['h_Wick_' + str(x)] = [5.0]
				else:
					newDat['h_Wick_' + str(x)] = [candles[x].getHighWick() / candles[x-1].getHighWick()]
				if candles[x-1].getLowWick() == 0:
					newDat['l_Wick_' + str(x)] = [5.0]
				else:
					newDat['l_Wick_' + str(x)] = [candles[x].getLowWick() / candles[x-1].getLowWick()]

				# Get intraday relative sizes
				if candles[x].getBody() == 0:
					newDat['h_Wick_i_' + str(x)] = [5.0]
				else:
					newDat['h_Wick_i_' + str(x)] = [candles[x].getHighWick() / candles[x].getBody()]
				if candles[x].getBody() == 0:
					newDat['l_Wick_i_' + str(x)] = [5.0]
				else:
					newDat['l_Wick_i_' + str(x)] = [candles[x].getLowWick() / candles[x].getBody()]

			# Determine label of data
			# If trend is rising
			if candles[0].get('Trend') == 1:
				# And future is also rising, bullish continuation
				if data.Close[cursor + 6] - candles[0].get('Open') > 0.0:
					newDat['Label'] = [0.0]
				# Otherwise, bearish reversal
				else:
					newDat['Label'] = [3.0]
			# Otherwise, trend is falling
			else:
				# But future is rising, bullish reversal
				if data.Close[cursor + 6] - candles[0].get('Open') > 0.0:
					newDat['Label'] = [2.0]
				# Otherwise, bearish continuation
				else:
					newDat['Label'] = [1.0]

			# For now, set label confidence to 1.0
			newDat['LabelConf'] = [1.0]

			# Remove out of range candle
			candles.pop(0)

		# Append data to either train or test depending on data division
		procDat = procDat.append(newDat)

	return procDat

def cluster(data):
	log.debug('Clustering Data')
	labels_true = data[['Label']]
	features = data.drop(columns=['Label', 'LabelConf'])

	labels_true = labels_true.to_numpy().flatten()
	features = features.to_numpy()

	print('-------------------> e, m value: ', e, ', ', m)
	db = DBSCAN(eps = 0.01*e, min_samples = 5*m).fit(features)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Get number of clusters in result
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0);
	n_noise_ = list(labels).count(-1)

	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
	print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
	print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
	print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
	print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
	# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(features, labels))

def evalDecisionTree(data, maxDepth, trials=1):
	log.debug('Evaluating Decision Tree Model')
	labels = data[['Label']]
	features = data.drop(columns=['Label', 'LabelConf'])

	labels = labels.to_numpy()
	features = features.to_numpy()

	results = []
	times = []
	for t in range(trials):
		log.debug('Trial %s', t)
		train_time = time.time()
		X_train, X_test, y_train, y_test = tts(features, labels, test_size = 0.2, shuffle = 1)
	
		model = dtc(max_depth = maxDepth).fit(X_train, y_train)
		times.append(time.time() - train_time)

		prediction = model.predict(X_test)
		accuracy = model.score(X_test, y_test)
		results.append(accuracy)

	avgtime = sum(times) / len(times);
	rate = sum(results) / len(results)
	return rate, avgtime

def evalRandomForest(data, maxDepth=2, min_samples_leaf=2, trials=1):
	log.debug('Evaluating Random Forest Model')
	labels = data[['Label']]
	features = data.drop(columns=['Label', 'LabelConf'])

	labels = labels.to_numpy().flatten()
	features = features.to_numpy()

	results = []
	times = []
	for t in range(trials):
		log.debug('Trial %s', t)
		train_time = time.time()
		X_train, X_test, y_train, y_test = tts(features, labels, test_size = 0.2, shuffle = 1)
	
		model = RandomForestClassifier(max_depth = maxDepth, random_state=0).fit(X_train, y_train)
		times.append(time.time() - train_time)

		prediction = model.predict(X_test)
		accuracy = model.score(X_test, y_test)
		results.append(accuracy)

	avgtime = sum(times) / len(times);
	rate = sum(results) / len(results)
	return rate, avgtime

def evalBaggingClassifier(data, n_estimators, trials=1):
	log.debug('Evaluating Random Forest Model')
	labels = data[['Label']]
	features = data.drop(columns=['Label', 'LabelConf'])

	labels = labels.to_numpy().flatten()
	features = features.to_numpy()

	results = []
	times = []
	for t in range(trials):
		log.debug('Trial %s', t)
		train_time = time.time()
		X_train, X_test, y_train, y_test = tts(features, labels, test_size = 0.2, shuffle = 1)
	
		model = BaggingClassifier(base_estimator=SVC(kernel='linear'), n_estimators=10, random_state=0).fit(X_train, y_train)
		times.append(time.time() - train_time)

		prediction = model.predict(X_test)
		accuracy = model.score(X_test, y_test)
		results.append(accuracy)

	avgtime = sum(times) / len(times);
	rate = sum(results) / len(results)
	return rate, avgtime

def evalSVM(data, kern, deg=0, trials=1):
	log.debug('Evaluating SVM Model')
	labels = data[['Label']]
	features = data.drop(columns=['Label', 'LabelConf'])

	labels = labels.to_numpy()
	features = features.to_numpy()
	
	results = []
	times = []
	for t in range(trials):
		log.debug('Trial %s', t)
		train_time = time.time()
		X_train, X_test, y_train, y_test = tts(features, labels, test_size = 0.2, shuffle = 1)
		y_train = np.ravel(y_train)
		y_test = np.ravel(y_test)
	
		model = SVC(kernel= kern, degree= deg).fit(X_train, y_train)
		times.append(time.time() - train_time)

		prediction = model.predict(X_test)
		accuracy = model.score(X_test, y_test)
		results.append(accuracy)

	avgtime = sum(times) / len(times);
	rate = sum(results) / len(results)
	return rate, avgtime

def evalKNN(data, neighbs, trials=1):
	log.debug('Evaluating KNN Model')
	labels = data[['Label']]
	features = data.drop(columns=['Label', 'LabelConf'])

	labels = labels.to_numpy()
	features = features.to_numpy()
	
	results = []
	for t in range(trials):
		log.debug('Trial %s', t)
		train_time = time.time()
		X_train, X_test, y_train, y_test = tts(features, labels, test_size = 0.2, shuffle = 1)
		y_train = np.ravel(y_train)
		y_test = np.ravel(y_test)
	
		model = KNeighborsClassifier(n_neighbors= neighbs).fit(X_train, y_train)
		log.debug('\tModel trained in %s seconds', time.time() - train_time)

		prediction = model.predict(X_test)
		accuracy = model.score(X_test, y_test)
		results.append(accuracy)

	rate = sum(results) / len(results)
	return rate

def evalNB(data, trials=1):
	log.debug('Evaluating Naive Bayes Model')
	labels = data[['Label']]
	features = data.drop(columns=['Label', 'LabelConf'])

	labels = labels.to_numpy()
	features = features.to_numpy()
	
	results = []
	times = []
	for t in range(trials):
		log.debug('Trial %s', t)
		train_time = time.time()
		X_train, X_test, y_train, y_test = tts(features, labels, test_size = 0.2, shuffle = 1)
		y_train = np.ravel(y_train)
		y_test = np.ravel(y_test)
	
		model = GaussianNB().fit(X_train, y_train)
		times.append(time.time() - train_time)

		prediction = model.predict(X_test)
		accuracy = model.score(X_test, y_test)
		results.append(accuracy)

	avgtime = sum(times) / len(times);
	rate = sum(results) / len(results)
	return rate, avgtime

def evalAll(data):
	# log.info('Decision Tree Depth 5:\t%s', evalDecisionTree(data, 5, 20))
	log.info('Decision Tree Depth 5:\t%s', evalDecisionTree(data, 10, 20))
	# log.info('Decision Tree Depth 10:\t%s', evalDecisionTree(data, 10, 20))
	# log.info('Decision Tree Depth 25:\t%s', evalDecisionTree(data, 25, 20))
	# log.info('Linear SVM: %s', evalSVM(data, 'linear', 20))
	# log.info('Radial Basis Kernel SVM: %s', evalSVM(data, 'rbf', 20))
	# log.info('Polynomial SVM degree 2: %s', evalSVM(data, 'poly', 2, 20))
	# log.info('Polynomial SVM degree 3: %s', evalSVM(data, 'poly', 3, 20))
	# log.info('Polynomial SVM degree 4: %s', evalSVM(data, 'poly', 4, 20))
	# log.info('KNN 4: %s', evalKNN(data, 4, 20))
	# log.info('KNN 8: %s', evalKNN(data, 8, 20))
	# log.info('KNN 12: %s', evalKNN(data, 12, 20))
	# log.info('KNN 20: %s', evalKNN(data, 20, 20))
	# log.info('KNN 40: %s', evalKNN(data, 40, 20))
	# log.info('Gaussian Naive Bayes: %s', evalNB(data, 20))
	log.info('Random Forest: %s', evalRandomForest(data, 10, 50, 20))
	# log.info('Random Forest: %s', evalRandomForest(data, 10, 1))
	# log.info('Bagging Classifier: %s', evalBaggingClassifier(data, 10, 20))

def main(argv):
	# Operational Arguments
	getNewData = False
	showGraphs = False
	obtNewPattern = False
	consolidate = True
	patternDays = 3
	trainPercentage = 0.8
	# Diversified
	# tickers = ['PG', 'JNJ', 'MMM', 'BRK-A', 'GOOGL', 'GE', 'DIS', 'DHR', 'HON','BHP', 'V', 'WMT', 'NSRGY', 'KO', 'NEE', 'ENLAY', 
	# 'XOM', 'CVX', 'AAPL', 'MSFT', 'TSM', 'AMZN', 'TSLA', 'BABA']
	tickers  = ['MSFT']

	log.basicConfig(level=log.INFO, format="%(levelname)s: %(message)s")

	# Commandline argument parsing
	# ====================================================
	try:
		opts, args = getopt.getopt(argv, 'ncogd:p:v', ['getNewData', 'consolidated' 'obtNewPattern', 'showgraphs', 'patternDays', 'trainPercentage', 'verbose'])
		for opt, arg in opts:
			if opt in ('-n', '--getNewData'):
				getNewData = True
				obtNewPattern = True
			if opt in ('-c', '--consolidated'):
				consolidate = True
			if opt in ('-o', '--obtNewPattern'):
				obtNewPattern = True
			if opt in ('-g', '--showgraphs'):
				showGraphs = True
			if opt in ('-d', '--patternDays'):
				patternDays = float(arg)
			if opt in ('-p', '--trainPercentage'):
				trainPercentage = float(arg)
			if opt in ('-v', '--verbose'):
				log.basicConfig(level=log.DEBUG, format="%(levelname)s: %(message)s")

	except getopt.GetoptError:
		log.debug('Incorrect usage')

	log.info('======================================')
	log.info('Begin Processing')
	log.info('======================================')

	# Acquire data for each requested ticker symbol
	# ====================================================
	total_time = time.time()
	raw = {}
	for ticker in tickers:
		# Retrieve data from API if requested or from saved data otherwise 
		if getNewData == True:
			log.info('Retrieving Data for %s', ticker)
			raw[ticker] = newData(ticker)
		else:
			try:
				log.info('Loading Data %s', ticker)
				raw[ticker] = pd.read_csv(ticker + '_raw.csv')
			except:
				log.error('Missing data for: %s (Data not yet retrieved?)', ticker)
				quit()
		log.info('Acquired Data with %s elements', len(raw[ticker].index))

		# If requested, show example graphs of candlesticks
		if showGraphs == True:
			raw[ticker].index = pd.to_datetime(raw[ticker].index)
			mpf.plot(raw[ticker], type='candlestick', title=ticker)

	log.info('------')

	# Retrieve processed data from file or raw data
	# ==========================================
	procDat = {}
	for ticker in tickers:
		if obtNewPattern == True:
			log.info('Obtaining Pattern Data for %s', ticker)
			data_time = time.time()
			procDat[ticker] = acqPatterns(raw[ticker], patternDays, trainPercentage) 
			log.info('Obtained %s total data points in %s seconds', len(procDat[ticker].index), time.time() - data_time)
		else:
			try:
				log.info('Loading Pattern Data for %s', ticker)
				procDat[ticker] = pd.read_csv(ticker + '_procDat.csv')
			except:
				log.error('Missing pattern data for: %s (Patterns not processed?)', ticker)
				quit()

		# Save unique ticker data
		procDat[ticker].to_csv(ticker + '_procDat.csv', index=False)

	# Train and test processed data
	if consolidate == True:
		consDat = newDataFrame(patternDays)
		for dat in procDat.values():
			consDat = consDat.append(dat)
		# Save consolidated data 
		consDat.to_csv('consolidated_procDat.csv', index=False)

		# Train consolidated data
		log.info('======================================')
		log.info('Resulting Accuracies for Consolidated')
		log.info('======================================')
		evalAll(consDat)
	else:
		for ticker in tickers:
			log.info('======================================')
			log.info('Resulting Accuracies for %s', ticker)
			log.info('======================================')
			evalAll(procDat[ticker])

	log.info('=================================')
	log.info('Finished in: %s', time.time() - total_time)

if __name__ == "__main__":
	main(sys.argv[1:])