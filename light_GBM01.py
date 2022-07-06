from google.colab import drive
drive.mount('/content/drive')

cd /content/drive/MyDrive/programming/uec/linear_regression/property_price_prediction/

!pip install -q rasterio rasterstats geopandas lightgbm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import time
 
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = pd.read_csv('DC_train.csv')
test = pd.read_csv('DC_test.csv')

train.dtypes

#使う特徴量を洗い出す。
X = train[['BATHRM','HF_BATHRM','NUM_UNITS','ROOMS','GBA','KITCHENS','LIVING_GBA']]
y = train[['PRICE']]
X.info()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)

lgb_train =lgb.Dataset(X_train,y_train)
lgb_test = lgb.Dataset(X_test,y_test)

#パラメーターの設定
params = {
    'random_state':1234, 'verbose':0,'metrics':'rmse'
}
num_round =100

#モデル訓練
model = lgb.train(params, lgb_train, num_boost_round = num_round)
#予測
prediction_LG = model.predict(y)
#少数丸め
prediction_LG = np.round(prediction_LG)

#submit用のデータをつくる
X_for_submit = test[['BATHRM','HF_BATHRM','NUM_UNITS','ROOMS','GBA','KITCHENS','LIVING_GBA']]
submit = test[['Id']]
#IdとPRICEが入ったsubmitデータが作れた
submit['PRICE'] =model.predict(X_for_submit)
submit.to_csv('/content/drive/MyDrive/programming/uec/linear_regression/property_price_prediction/submission/submit04.csv',index=False)