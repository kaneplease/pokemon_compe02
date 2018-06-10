# パッケージのインポート
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report
# matplotlibのインポートとおまじない
#import matplotlib.pyplot as plt

#csvの読み込み
type_list = pd.read_csv('./type_list_2.csv', index_col=0)
pokemon_list = pd.read_csv('./pokemon.csv', index_col=0).drop(["Name","Generation"],axis=1).fillna("Nothing")
train_data = pd.read_csv('./train.csv', index_col=0)
test_data = pd.read_csv('./test.csv', index_col=0)

#テストか判定
test_num = 0

#勝敗を0,1で格納、これがラベルになる
train_result = np.zeros(len(train_data.index))
#入力データ場所の確保
input_data = np.zeros((len(train_data.index), 7))
input_data_test = np.zeros((len(test_data.index), 7))

for i in range(len(train_data.index)):
    #ラベルの作成
    y_dum = (train_data.iloc[i,0] == train_data.iloc[i,2])
    train_result[i] = (1 if y_dum else 0)

    #入力の作成
    #print(train_data.iloc[i,0])
    first_pokemon = pokemon_list.iloc[(train_data.iloc[i,0]-1)]
    second_pokemon = pokemon_list.iloc[(train_data.iloc[i,1]-1)]

    type_mag = np.ones((2,4))
    type_mag[0][0] = type_list.loc[first_pokemon["Type 1"], second_pokemon["Type 1"]]
    type_mag[0][1] = type_list.loc[first_pokemon["Type 1"], second_pokemon["Type 2"]]
    type_mag[0][2] = type_list.loc[first_pokemon["Type 2"], second_pokemon["Type 1"]]
    type_mag[0][3] = type_list.loc[first_pokemon["Type 2"], second_pokemon["Type 2"]]

    type_mag[1][0] = type_list.loc[second_pokemon["Type 1"], first_pokemon["Type 1"]]
    type_mag[1][1] = type_list.loc[second_pokemon["Type 1"], first_pokemon["Type 2"]]
    type_mag[1][2] = type_list.loc[second_pokemon["Type 2"], first_pokemon["Type 1"]]
    type_mag[1][3] = type_list.loc[second_pokemon["Type 2"], first_pokemon["Type 2"]]

    #print(first_pokemon)
    mag_f_to_s = type_mag[0][0] * type_mag[0][1] * type_mag[0][2] * type_mag[0][3]
    mag_s_to_f = type_mag[1][0] * type_mag[1][1] * type_mag[1][2] * type_mag[1][3]

    input_data[i][0] = first_pokemon["Attack"] / second_pokemon["Defense"] * mag_f_to_s
    input_data[i][1] = second_pokemon["Attack"] / first_pokemon["Defense"] * mag_s_to_f
    input_data[i][2] = first_pokemon["Sp. Atk"] / second_pokemon["Sp. Def"] * mag_f_to_s
    input_data[i][3] = second_pokemon["Sp. Atk"] / first_pokemon["Sp. Def"] * mag_s_to_f
    input_data[i][4] = first_pokemon["HP"] - second_pokemon["HP"]
    input_data[i][5] = first_pokemon["Speed"] - second_pokemon["Speed"]
    input_data[i][6] = (1 if first_pokemon["Legendary"] else 0) - (1 if second_pokemon["Legendary"] else 0)

    ##################################
for i in range(len(test_data.index)):
    #テストデータも整形する
    first_pokemon_test = pokemon_list.iloc[(test_data.iloc[i, 0] - 1)]
    second_pokemon_test = pokemon_list.iloc[(test_data.iloc[i, 1] - 1)]

    type_mag_test = np.ones((2, 4))
    type_mag_test[0][0] = type_list.loc[first_pokemon_test["Type 1"], second_pokemon_test["Type 1"]]
    type_mag_test[0][1] = type_list.loc[first_pokemon_test["Type 1"], second_pokemon_test["Type 2"]]
    type_mag_test[0][2] = type_list.loc[first_pokemon_test["Type 2"], second_pokemon_test["Type 1"]]
    type_mag_test[0][3] = type_list.loc[first_pokemon_test["Type 2"], second_pokemon_test["Type 2"]]

    type_mag_test[1][0] = type_list.loc[second_pokemon_test["Type 1"], first_pokemon_test["Type 1"]]
    type_mag_test[1][1] = type_list.loc[second_pokemon_test["Type 1"], first_pokemon_test["Type 2"]]
    type_mag_test[1][2] = type_list.loc[second_pokemon_test["Type 2"], first_pokemon_test["Type 1"]]
    type_mag_test[1][3] = type_list.loc[second_pokemon_test["Type 2"], first_pokemon_test["Type 2"]]

    mag_f_to_s_test = type_mag_test[0][0] * type_mag_test[0][1] * type_mag_test[0][2] * type_mag_test[0][3]
    mag_s_to_f_test = type_mag_test[1][0] * type_mag_test[1][1] * type_mag_test[1][2] * type_mag_test[1][3]

    input_data_test[i][0] = first_pokemon_test["Attack"] / second_pokemon_test["Defense"] * mag_f_to_s_test
    input_data_test[i][1] = second_pokemon_test["Attack"] / first_pokemon_test["Defense"] * mag_s_to_f_test
    input_data_test[i][2] = first_pokemon_test["Sp. Atk"] / second_pokemon_test["Sp. Def"] * mag_f_to_s_test
    input_data_test[i][3] = second_pokemon_test["Sp. Atk"] / first_pokemon_test["Sp. Def"] * mag_s_to_f_test
    input_data_test[i][4] = first_pokemon_test["HP"] - second_pokemon_test["HP"]
    input_data_test[i][5] = first_pokemon_test["Speed"] - second_pokemon_test["Speed"]
    input_data_test[i][6] = (1 if first_pokemon_test["Legendary"] else 0) - (1 if second_pokemon_test["Legendary"] else 0)


df = pd.DataFrame(input_data, columns=list('ABCDEFG'))
df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

df_test = pd.DataFrame(input_data_test, columns=list('ABCDEFG'))
df_test = df_test.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

df1 = df[:int(len(df)/2)]
df2 = df[int(len(df)/2):len(df)]
train_result1 = train_result[:int(len(df)/2)]
train_result2 = train_result[int(len(df)/2):len(df)]

#dfが変数、train_resultがラベル
# モデルのインスタンス作成

mod = xgb.XGBRegressor(objective='binary:logistic', eta = 0.01, max_depth=7, subsample=0.91)

if test_num == 1:
    mod.fit(df1, train_result1)

    pred_xgb = pd.Series(mod.predict(df2))

    print(pred_xgb)
    logloss = log_loss(train_result2, pred_xgb, eps=1e-15)
    print(logloss)

if test_num != 1:
    mod.fit(df, train_result)

    pred_xgb = pd.Series(mod.predict(df_test))
    pred_xgb.to_csv("pred_xgb_3.csv")
    print("END")


"""
for i in range(100):
    print(pred_xgb[i],".vs.",train_result2[i])
"""


