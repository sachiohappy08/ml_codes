!pip install japanize-matplotlib

# ライブラリの読み込み
import os
import scipy as sp
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib #日本語化matplotlib
import seaborn as sns
sns.set(font="IPAexGothic") #日本語フォント設定
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
# グラフをインラインで表示させる
%matplotlib inline

train = pd.read_csv('/content/drive/MyDrive/programming/uec/linear_regression/property_price_prediction/DC_train.csv')
test = pd.read_csv('/content/drive/MyDrive/programming/uec/linear_regression/property_price_prediction/DC_test.csv')

#データ型と欠損値の確認
train.dtypes
train.isnull().sum()
test.dtypes
test.isnull().sum()

#intとfloatだけで、nullがないものだけでとりあえず特徴量作る
X = train[['BATHRM','BATHRM','ROOMS','BEDRM','FIREPLACES','LANDAREA']]
y = train[['PRICE']]

from sklearn.model_selection import train_test_split
#splitをする
X_train, X_test, y_train,y_test = train_test_split(X,y,random_state=0)

#k近傍法を行ってみる。まずtrainデータでモデルをfitさせる。
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit(X_train,y_train)

#上で作ったモデルをテストデータに当てはめてどのぐらいあっているか確かめる。めちゃ精度低いw
knn.score(X_test,y_test)

#submit用のデータをつくる
X_for_submit = test[['BATHRM','BATHRM','ROOMS','BEDRM','FIREPLACES','LANDAREA']]
submit = test[['Id']]
#IdとPRICEが入ったsubmitデータを作った
submit['PRICE'] = knn.predict(X_for_submit)

#submit用のファイルを作る。indexはFalseにしないとkaggleでエラーになる
submit.to_csv('hogehoge/submit01.csv',index=False)