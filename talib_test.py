import numpy as np
import pandas as pd 
import talib 
import sys
import inspect

from talib.abstract import *


dim = 1000
T1 = 20
T2 = 10
T3 = 7


data_toy = {
    'open': np.random.rand(dim),
    'high': np.random.rand(dim),
    'low': np.random.rand(dim),
    'close': np.random.rand(dim),
    'volume': np.random.rand(dim),
    'periods': np.array([1.0]*dim)
}


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


data = data_toy
features = {}


for indicator in talib.get_functions():
	function_indicator = getattr(talib.abstract, indicator)
	features[indicator] = function_indicator(data_toy, **extra)


print(features['T3'])
