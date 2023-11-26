from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
from tqdm import tqdm_notebook
import numpy as np

def best_model_training(X_train, y_train, X_test, y_test, model, param_grid):
    rf = model()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=3, n_jobs=-1)

    # Create a progress bar
    with tqdm_notebook(total=len(list(ParameterGrid(param_grid))), desc="Grid Search Progress") as bar:
        def update_progress(_):
            bar.update()

        # Override the progress_callback to update the progress bar
        grid_search.progress_callback = update_progress

        grid_search.fit(X_train, y_train)

    # Best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("------------Model Assessment------------\n")
    print(f"MAE, error: {metrics.mean_absolute_error(y_test, y_pred)}")
    print(f"MSE, error: {metrics.mean_squared_error(y_test, y_pred)}")
    print(f"RMSE, error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")
    print(f"r2: {metrics.r2_score(y_test, y_pred)}")
    print("\n")

    return best_model