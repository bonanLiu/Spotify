import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
import time

from A_clean import data_df

## train and test

x=data_df[['energy','loudness','explicit','danceability','tempo','duration_ms','instrumentalness','acousticness','liveness']]
y=data_df['popularity']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.15, random_state=42)



##  XGBoost

# from xgboost import XGBRegressor
# from C_classification import param_grid_XGB

# start_XGB = time.time()
# random_search_XGB = RandomizedSearchCV(
#     estimator=XGBRegressor(random_state=42),
#     param_distributions=param_grid_XGB,
#     n_iter=50, 
#     scoring='neg_mean_squared_error',
#     cv=3,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )
# random_search_XGB.fit(x_train, y_train)

# best_XGB = random_search_XGB.best_estimator_
# XGB_pre = best_XGB.predict(x_test)

# mse_XGB = mean_squared_error(y_test, XGB_pre)
# rmse_XGB = np.sqrt(mse_XGB)
# mae_XGB = mean_absolute_error(y_test, XGB_pre)
# r2_XGB = r2_score(y_test, XGB_pre)

# print("Best Parameters:")
# for param, val in random_search_XGB.best_params_.items():
#     print(f"  {param}: {val}")

# print("\nEvaluation Metrics:")
# print(f"MSE:  {mse_XGB:.4f}")
# print(f"RMSE: {rmse_XGB:.4f}")
# print(f"MAE:  {mae_XGB:.4f}")
# print(f"R²:   {r2_XGB:.4f}")

# end_XGB = time.time()
# print(f"\n⏱️ Training time: {end_XGB - start_XGB:.5f} seconds")





## LinearRegression
from sklearn.linear_model import LinearRegression

param_grid_LR = {
    'fit_intercept': [True, False],
    'positive': [False, True],
    'n_jobs': [None, -1]
}

start_LR = time.time()

grid_search_LR = GridSearchCV(
    estimator=LinearRegression(),
    param_grid=param_grid_LR,
    scoring='neg_mean_squared_error', 
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search_LR.fit(x_train, y_train)

best_LR = grid_search_LR.best_estimator_
LR_pred = best_LR.predict(x_test)

mse_LR = mean_squared_error(y_test, LR_pred)
rmse_LR = np.sqrt(mse_LR)
mae_LR = mean_absolute_error(y_test, LR_pred)
r2_LR = r2_score(y_test, LR_pred)

print("\nBest Parameters:")
for param, val in grid_search_LR.best_params_.items():
    print(f"  {param}: {val}")

print("\nEvaluation Metrics (Linear Regression):")
print(f"MSE:  {mse_LR:.4f}")
print(f"RMSE: {rmse_LR:.4f}")
print(f"MAE:  {mae_LR:.4f}")
print(f"R²:   {r2_LR:.4f}")

end_LR = time.time()
print(f"\n⏱️ Training time: {end_LR - start_LR:.5f} seconds")



# print("\nFeature Coefficients:")
# for name, coef in zip(x_train.columns, best_LR.coef_):
#     print(f"{name}: {coef:.4f}")





# ## RandomForest - Regression
# from sklearn.ensemble import RandomForestRegressor
# from C_classification import param_grid_RF

# start_RF = time.time()

# random_search_RF = RandomizedSearchCV(
#     estimator=RandomForestRegressor(random_state=42),
#     param_distributions=param_grid_RF,
#     n_iter=30,
#     scoring='neg_mean_squared_error',
#     cv=3,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )

# random_search_RF.fit(x_train, y_train)

# best_RF = random_search_RF.best_estimator_
# RF_pred = best_RF.predict(x_test)

# mse_RF = mean_squared_error(y_test, RF_pred)
# rmse_RF = np.sqrt(mse_RF)
# mae_RF = mean_absolute_error(y_test, RF_pred)
# r2_RF = r2_score(y_test, RF_pred)

# print("\nBest Parameters:")
# for param, val in random_search_RF.best_params_.items():
#     print(f"  {param}: {val}")

# print("\nEvaluation Metrics (RandomForestRegressor):")
# print(f"MSE:  {mse_RF:.4f}")
# print(f"RMSE: {rmse_RF:.4f}")
# print(f"MAE:  {mae_RF:.4f}")
# print(f"R²:   {r2_RF:.4f}")

# end_RF = time.time()
# print(f"\n⏱️ Training time: {end_RF - start_RF:.5f} seconds")




