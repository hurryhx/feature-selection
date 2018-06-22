import numpy as np
import pandas as pd 
import talib 
import sys
import inspect

from talib.abstract import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing


T1_list = [6,10,20,30,40,60,80,100]
T2 = 10
T3 = 7


# raw_data = pd.read_csv('minute/df_ohlc_minute.csv')
raw_data = pd.read_csv('minute/df_ohlc_5minute.csv')
raw_num_col = raw_data.shape[0]
datetime = raw_data.iloc[:,0]


## cyclic encoding
hour = np.sin(np.array(datetime.map(lambda x: int(x[11:13])))*np.pi/24)
month = np.sin(np.array(datetime.map(lambda x: int(x[5:7]))))


## orginal OHLCV needed for talib functions
x_original = {
	'open': np.array(raw_data['open']),
	'high': np.array(raw_data['high']),
	'low': np.array(raw_data['low']),
	'close': np.array(raw_data['close']),
	'volume': np.array(raw_data['volume']),
	'periods': np.array([1.0]*raw_num_col),
}


## extra params needed for talib functions

def make_extra(T1, T2, T3):

	extra = {
		'timeperiod': T1,
		'fastperiod': 0.5*T1,
		'slowperiod': 1.5*T1,
		'signalperiod': 0.3*T1,
		'fastk_period': 0.5*T2, 
		'fastd_period': 0.3*T2,
		'slowk_period': 0.3*T2,
		'slowd_period': 0.3*T2,
		'timeperiod1': T3,
		'timeperiod2': 2*T3,
		'timeperiod3': 4*T3,

		'matype': 0,
		'fastmatype': 0,
		'slowmatype': 0,
		'fastd_matype': 0,
		'fastk_matype': 0,
		'slowk_matype': 0,
		'slowd_matype': 0,

		'nbdevup': 2,
		'nbdevdn': 2,
		'fastlimit': 0.5,
		'slowlimit': 0.01,
		'minperiod': 0.2*T2,
		'maxperiod': 3*T2,
		'accelaration': 0,
		'maximum': 0,
		'startvalue': 0,
		'offsetonreverse': 0,
		'accelerationinitlong': 0,
		'accelerationlong': 0,
		'accelerationmaxlong': 0,
		'accelerationinitshort': 0,
		'accelerationshort': 0,
		'accelerationmaxshort': 0,
		'vfactor': 0,

		'penetration': 0,

		'nbdev': 1
	}

	return extra




features = {}

loop_number = len(T1_list)
indicator_number = 171

x_matrix = np.zeros(shape=(indicator_number*loop_number,raw_num_col))

features['open'] = x_matrix[0] = np.array(raw_data['open'])
features['high'] = x_matrix[1] = np.array(raw_data['high'])
features['low'] = x_matrix[2] = np.array(raw_data['low'])
features['close'] = x_matrix[3] = np.array(raw_data['close'])
features['volume'] = x_matrix[4] = np.array(raw_data['volume'])
features['hour'] = x_matrix[5] = hour

features_name_list = list(raw_data.columns)[1:]
features_name_list.append('hour')

idx = len(features_name_list)


for T1 in T1_list:

	extra = make_extra(T1, T2, T3)

	for indicator in talib.get_functions():

		excluded_list = ['ASIN','ACOS','CEIL','COSH','EXP','FLOOR','LOG10','SINH','TAN','TANH']

		if indicator not in excluded_list:

			function_indicator = getattr(talib.abstract, indicator)
			result = function_indicator(x_original, **extra)

			features[indicator] = result
			

			if len(result) != raw_num_col:
				x_matrix[idx:idx+len(result)] = np.array(result)
				idx += len(result)

				for i in range(len(result)):
					features_name_list.append(indicator+'T'+str(T1)+'S'+str(i+1))

			else:
				x_matrix[idx] = np.array(result)
				features_name_list.append(indicator+'T'+str(T1))
				idx += 1



x_matrix = np.swapaxes(x_matrix,0,1)
shape0_original = x_matrix.shape[0]
# print(x_matrix.shape)

x_matrix[np.isneginf(x_matrix)] = np.nan
x_matrix[np.isinf(x_matrix)] = np.nan

x_matrix = x_matrix[~np.isnan(x_matrix).any(axis=1)]
shape0_after = x_matrix.shape[0]
num_na = shape0_original - shape0_after
# print(x_matrix.shape)

x_matrix = x_matrix[1:,:]
# print(x_matrix.shape)

close = x_original['close']
y_array = np.array([a1 / a2 - 1.0 for a1, a2 in zip(close[num_na+1:], close[num_na:])])
# print(y_array.shape)

# print(features_name_list)


### Discard old noisy datas

X = x_matrix[200000:,:]
Y = y_array[200000:]



# print(features_name_list[0])
# print(train_X[10,:])

# print(features_name_list[59])
# print(features_name_list[60])
# print(features_name_list[65])




# pre-defined bins labeling
def y_labeling(y, bins):

	return np.digitize(y, bins)


bins = [-3e-3, -1e-3, +1e-3, +3e-3]
Y = y_labeling(Y,bins)


## Train-Test Splitting
X = preprocessing.scale(X)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.1, shuffle=False)

# print(train_X[0,:])
# print(train_X[1,:])
# print(train_X[2,:])
# print(train_X[3,:])
# print(train_X[4,:])
# print(train_Y[:100])
# print(train_Y[-100:])


print('Training Features Shape:', train_X.shape)
print('Training Labels Shape:', train_Y.shape)
print('Testing Features Shape:', test_X.shape)
print('Testing Labels Shape:', test_Y.shape)
print(len(features_name_list))

# unique, counts = np.unique(Y, return_counts=True)
# print(dict(zip(unique, counts)))



## Baseline --- Linear Classifier SGD
SGD = SGDClassifier()

print ("Training Linear Classifier model.")
SGD.fit(train_X, train_Y)
# print (clf.coef_)
# print (clf.intercept_)

train_Y_pred = SGD.predict(train_X)
print ('Training Accuracy: ', metrics.accuracy_score(train_Y, train_Y_pred))

test_Y_pred = SGD.predict(test_X)
print ('Testing Accuracy: ', metrics.accuracy_score(test_Y, test_Y_pred))

print('--------------------------')


## Baseline --- SVM Very Slow Don't Know Why
# clf_2 = svm.SVC()

# print ("Training SVM model.")
# clf_2.fit(train_X, train_Y)

# train_Y_pred = clf_2.predict(train_X)
# print ('Training Accuracy: ', metrics.accuracy_score(train_Y, train_Y_pred))

# test_Y_pred = clf_2.predict(test_X)
# print ('Testing Accuracy: ', metrics.accuracy_score(test_Y, test_Y_pred))
# print('--------------------------')



## Random Forest
RF = RandomForestClassifier()

print ("Training Random Forest Classifier model.")
RF.fit(train_X, train_Y)

train_Y_pred = RF.predict(train_X)
print ('Training Accuracy: ', metrics.accuracy_score(train_Y, train_Y_pred))

test_Y_pred = RF.predict(test_X)
print ('Testing Accuracy: ', metrics.accuracy_score(test_Y, test_Y_pred))

# Feature Importance
importances = list(RF.feature_importances_)
feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(features_name_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
print('--------------------------')












