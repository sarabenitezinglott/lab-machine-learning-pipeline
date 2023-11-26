import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def split_65(X,y): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    return X_train, X_test, y_train, y_test

regressor_models_op = {"LinReg": LinearRegression(), 
                 "LinRidge": Ridge(),
                 "LinLasso" : Lasso(), 
                 "SGD": SGDRegressor(),
                 "knn": KNeighborsRegressor(),
                 "grad": GradientBoostingRegressor(),
                 "svr": SVR(),
                 "rf": RandomForestRegressor()}
                 


def training(X_train, y_train): 
    global regressor_models_op
    for model_name, model in regressor_models_op.items():
        print(f"Training: {model_name}")
        model.fit(X_train, y_train)
        rank = {}
        if hasattr(model, 'feature_importances_'):
            np.argsort(model.feature_importances_)
            for i, j in zip(model.feature_importances_, X_train.columns):
                rank[j] = i
            print(rank)
        elif hasattr(model, 'coef_'):
            np.argsort(model.coef_)
            for i, j in zip(model.coef_, X_train.columns):
                rank[j] = i
            print(rank)
        elif hasattr(model, 'ranking_'):
            np.argsort(model.ranking_)
            for i, j in zip(model.ranking_, X_train.columns):
                rank[j] = i
            print(rank)
        else:                                      #For svr and knn as they do this internally
            print(f"Feature ranking not available for {model_name}")

        
def assess_models(X_test, y_test):
    for model_name, model in regressor_models_op.items():
        y_pred = model.predict(X_test)
        print(f"------------{model_name}------------\n")
        print(f"MAE, error: {metrics.mean_absolute_error(y_test, y_pred)}")
        print(f"MSE, error: {metrics.mean_squared_error(y_test, y_pred)}")
        print(f"RMSE, error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")
        print(f"r2: {metrics.r2_score(y_test, y_pred)}")
        print("\n")