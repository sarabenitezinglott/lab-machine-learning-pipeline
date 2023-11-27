from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
import numpy as np

def gardient_training(X_train, y_train, X_test, y_test):
    #Hyperparameter tuning
    gradient_param = {
        'max_depth': [10, 15, 20], 
        'n_estimators': [100, 200, 500]}

    gradientgrid = GridSearchCV(GradientBoostingRegressor(random_state=79), gradient_param)
    gradientgrid.fit(X_train, y_train)
    print(gradientgrid.best_params_)
    print('Best score: ', gradientgrid.best_score_)

    gb_model = GradientBoostingRegressor(max_depth= 10, n_estimators= 100)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    print(f"MAE, error: {metrics.mean_absolute_error(y_test, y_pred_gb)}")
    print(f"MSE, error: {metrics.mean_squared_error(y_test, y_pred_gb)}")
    print(f"RMSE, error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred_gb))}")
    print(f"r2: {metrics.r2_score(y_test, y_pred_gb)}")


def forest_training(X_train, y_train, X_test, y_test):
    #Hyperparameter tuning
    randforest_param = {
        'max_depth': [10, 15, 20], 
        'n_estimators': [100, 200, 500]}

    forestgrid = GridSearchCV(RandomForestRegressor(random_state=79), randforest_param)
    forestgrid.fit(X_train, y_train)
    print(forestgrid.best_params_)
    print('Best score: ', forestgrid.best_score_)

    rf_model = RandomForestRegressor(max_depth= 15, n_estimators= 500)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print(f"MAE, error: {metrics.mean_absolute_error(y_test, y_pred_rf)}")
    print(f"MSE, error: {metrics.mean_squared_error(y_test, y_pred_rf)}")
    print(f"RMSE, error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))}")
    print(f"r2: {metrics.r2_score(y_test, y_pred_rf)}")