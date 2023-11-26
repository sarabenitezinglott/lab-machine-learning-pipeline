import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def import_data():
    df_train = pd.read_csv("../lab-machine-learning-pipeline/data/train.csv")
    df_test = pd.read_csv("../lab-machine-learning-pipeline/data/test.csv")
    return df_train, df_test

def exploration(df):
    print(df.info)
    print(df.isna().sum())
    print(df.describe)

def cleaning(df):
    #CARAT column
    df["carat"].max()
    df["carat"].min() 
    df["cut"].value_counts()
    #CUT class
    cut_class = {'Ideal':1,'Premium':2,'Very Good':3 ,'Good':4,'Fair':5}
    df["cut"] = df["cut"].replace(cut_class)
    #COLOR class
    df["color"].value_counts()
    color_class = {'D':1,'E':2,'F':3,'G':4,'H':5,'I':6,'J':7}
    df["color"] = df["color"].replace(color_class)
    #CLARITY class
    df["clarity"].value_counts()
    clarity_class = {'FL':1,'IF':2,'VVS1':3,'VVS2':4,'VS1':5,'VS2':6,'SI1':7,'SI2':8,'I1':9,'I2':10,'I3':11}
    df["clarity"] = df["clarity"].replace(clarity_class)
    #TOP width (Table)
    scalerx = MinMaxScaler()
    df["table"] = scalerx.fit_transform(df["table"].values.reshape(-1, 1))
    #LENGTH (X)
    scalerx = MinMaxScaler()
    df["x"] = scalerx.fit_transform(df["x"].values.reshape(-1, 1))
    #WIDTH (Y)
    scalery = MinMaxScaler()
    df["y"] = scalery.fit_transform(df["y"].values.reshape(-1, 1))
    #DEPTH (Z)
    scalerz = MinMaxScaler()
    df["z"] = scalerz.fit_transform(df["z"].values.reshape(-1, 1))
    #DEPTH PERCENTAGE
    scalerdepth = MinMaxScaler()
    df["depth"] = scalerdepth.fit_transform(df["depth"].values.reshape(-1, 1))

    return df


