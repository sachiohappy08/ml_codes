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

#使う特徴量を洗い出す。キッチン外してみる
X = train[['BATHRM','HF_BATHRM','NUM_UNITS','ROOMS','GBA','LIVING_GBA']]
y = train[['PRICE']]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123)

lgb_train =lgb.Dataset(X_train,y_train)
lgb_test = lgb.Dataset(X_test,y_test)
# #評価基準を決める
params = {'metric' :'rmse'}

#訓練データから回帰モデルを作成する。
gbm = lgb.train(params, lgb_train)

#テストデータを使って予測精度を確認する
test_predicted = gbm.predict(X_test)
predicted_df = pd.concat([y_test.reset_index(drop=True),pd.Series(test_predicted)],axis=1)
predicted_df.columns = ['true','predicted']

#予測値を図で確認する
def prediction_accuracy(predicted_df):
    RMSE = np.sqrt(mean_squared_error(predicted_df['true'], predicted_df['predicted']))
    plt.figure(figsize=(7,7))
    ax = plt.subplot(111)
    ax.scatter('true','predicted',data = predicted_df)
    ax.set_xlabel('True Price',fontsize=12)
    ax.set_ylabel('Predicted Price', fontsize=12)
    plt.tick_params(labelsize =15)
    x = np.linspace(20,50)
    y = x
    ax.plot(x,y, 'r-')
    plt.text(0.1,0.9, 'RMSE={}'.format(str(round(RMSE,3))),transform = ax.transAxes, fontsize=15)
prediction_accuracy(predicted_df)

lgb.plot_importance(gbm, height= 0.5, figsize = (8,16))

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
submit.to_csv('/content/drive/MyDrive/programming/uec/linear_regression/property_price_prediction/submission/submit05.csv',index=False)