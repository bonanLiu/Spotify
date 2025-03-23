from A_clean import data_df
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
## train and test


x=data_df[['valence', 'explicit', 'instrumentalness', 'danceability', 'loudness',
                         'acousticness', 'liveness', 'duration_ms', 'tempo', 'energy']]
y=data_df['popularity_new']
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)


# ## XGBoost

# from xgboost import XGBClassifier

# start_XGB = time.time()
param_grid_XGB = {
    'n_estimators': [50,100,200,500],
    'max_depth': [4,6,8,10],
    'learning_rate': [0.05,0.07,0.1],
    'min_child_weight': [1,3,5,7],
    'gamma': [0,0.1,0.3,0.5,1],
    'subsample': [0.5,0.7,0.8,1.0],
    'colsample_bytree': [0.5,0.7,0.8,1.0],
    'reg_alpha': [0,0.01,0.1],
    'reg_lambda': [0.1,0.5,1,5]
}

# random_search_XGB = RandomizedSearchCV(
#     estimator=XGBClassifier(eval_metric='logloss', random_state=42),
#     param_distributions=param_grid_XGB,
#     n_iter=50,              
#     scoring='accuracy',
#     cv=3,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )

# random_search_XGB.fit(x_train, y_train)

# end_XGB = time.time()
# print("Best Params:")
# for param, value in random_search_XGB.best_params_.items():
#     print(f"  {param}: {value}")
# print("Best Score:", random_search_XGB.best_score_)
# print(f"⏱️ Training time: {end_XGB-start_XGB:.5f} seconds")






# ## RandomForest

# from sklearn.ensemble import RandomForestClassifier

# start_RF = time.time()
param_grid_RF = {
    'n_estimators': [50,100,200,300],
    'max_depth': [5,10,15,20],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,5],
    'max_features': ['sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 50, 100]
}

# random_search_RF = RandomizedSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_distributions=param_grid_RF,
#     n_iter=50,              
#     scoring='accuracy',
#     cv=3,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )

# random_search_RF.fit(x_train, y_train)

# end_RF = time.time()
# print("Best Params:")
# for param, value in random_search_RF.best_params_.items():
#     print(f"  {param}: {value}")
# print("Best Score:", random_search_RF.best_score_)
# print(f"⏱️ Training time: {end_RF-start_RF:.5f} seconds")







