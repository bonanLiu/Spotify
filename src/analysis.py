import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from scipy.stats import zscore


data_df=pd.read_csv("data.csv")
# print(data_df['popularity'].unique())

data_df['popularity_new'] = np.where(data_df['popularity'].between(1, 50), 0, 1)
print(data_df['popularity_new'].value_counts())

## Reorganize the columns
data_df = data_df.drop(columns=['name', 'artists','release_date'])
data_df = data_df[[ 'id','year','valence', 'acousticness',  'danceability',
       'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'popularity_new']]

# ##  Basic Information
# print(data_df.info(), data_df.shape, data_df.describe())

# ## Data Cleaning
# print(data_df.isnull().sum())
# print(data_df.isna().sum())

# data_df=data_df.drop_duplicates()
# print(data_df.shape)

## Find Outliers - All features (x values)
ZS=zscore( data_df[['valence', 'acousticness',  'danceability',
       'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo']])
outliers=(abs(ZS>5).sum(axis=0))
outliersMark=abs(ZS)>5
outliersData=data_df[outliersMark.any(axis=1)]

data_df = data_df[(abs(ZS)<=5).all(axis=1)]
# print(data_df.shape)




##  Finding features

# 01. Pearson and Heatmap
data_df_p=data_df.drop(columns=['id','year'])
data_pearson=data_df_p.corr()
plt.figure(figsize=(20, 8))
sb.heatmap(data_pearson, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap of Red Wine Features')
# plt.show()

data_pearson=data_pearson.drop(columns=['valence', 'acousticness',  'danceability',
       'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo'])
data_pearson_f=data_pearson[data_pearson["popularity_new"] > 0]
# print(data_pearson_f)



# 02. PCA 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca_features=['valence', 'acousticness',  'danceability',
       'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo']

x_pcaChosen=data_df[pca_features]
scaler=StandardScaler()
x_scale=scaler.fit_transform(x_pcaChosen)

pcaTest=PCA(n_components=len(pca_features))
x_pcaTest=pcaTest.fit_transform(x_scale)

# Getting PCA result
pca_components_df=pd.DataFrame(x_pcaTest, columns=[f"PC{i+1}" for i in range(len(pca_features))])
pca_components_df["popularity_new"]=data_df["popularity_new"]

# Calculate the importance of each feature
correlation_with_popularity=pca_components_df.corr()["popularity_new"].drop("popularity_new").abs().sort_values(ascending=False)

# Top 5
top_5pca=correlation_with_popularity.index[:5]
load_all_pca=pd.DataFrame(pcaTest.components_.T, index=pca_features, columns=[f"PC{i+1}" for i in range(len(pca_features))])
feature_importance=load_all_pca[top_5pca].abs().sum(axis=1).sort_values(ascending=False)

# Into 3 group
Group01=list(feature_importance.index[:5])
Group02=list(feature_importance.index[5:10])
Group03=list(feature_importance.index[10:])

Group01 += [""] * (5 - len(Group01))  
Group02 += [""] * (5 - len(Group02))  
Group03 += [""] * (5 - len(Group03))

groups_features = pd.DataFrame({
    "Group 01 (Most Important)": Group01,
    "Group 02 (Moderate Importance)": Group02,
    "Group 03 (Less Important)": Group03[:5]
})

# print(groups_features)

# the features selected by PCA method 
#  Group 01 (Most Important) Group 02 (Moderate Importance) Group 03 (Less Important)
# 0                   valence                    speechiness          instrumentalness
# 1              danceability                    duration_ms                      mode
# 2                  loudness                         energy                       key
# 3              acousticness                          tempo
# 4                  explicit                       liveness


## Conclusion: Compare with the result in Pearson and PCA, we choose the 6 features to do the prediction



### Model

from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report

## train and test

x=data_df[['valence','explicit','danceability','loudness','acousticness']]
y=data_df['popularity_new']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.15, random_state=42,stratify=y)





# ##Decision Tree
# from sklearn.tree import DecisionTreeRegressor

# tree=DecisionTreeRegressor()
# tree.fit(x_train,y_train)
# tree_pre=tree.predict(x_test)
# mse_tree = mean_squared_error(y_test, tree_pre)
# rmse_tree = np.sqrt(mse_tree)
# mae_tree = mean_absolute_error(y_test, tree_pre)
# r2_tree = r2_score(y_test, tree_pre)

# print(mse_tree,rmse_tree,mae_tree,r2_tree)


# ## RandomForest - Regression
# from sklearn.ensemble import RandomForestRegressor
# RF=RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)  # change the max_depth from 5-8 or upper(but waste time) to boost the R2
# RF.fit(x_train,y_train)
# RF_pre=RF.predict(x_test)
# mse_RF = mean_squared_error(y_test, RF_pre)
# rmse_RF = np.sqrt(mse_RF)
# mae_RF = mean_absolute_error(y_test, RF_pre)
# r2_RF = r2_score(y_test, RF_pre)
# print(mse_RF,rmse_RF,mae_RF,r2_RF)


# ## RandomForest - Classifier
from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42,)
RFC.fit(x_train,y_train)
RFC_pre=RFC.predict(x_test)


accuracy_RFC=accuracy_score(y_test, RFC_pre)
print(f"Random Forest Accuracy: {accuracy_RFC:.4f}")
print(classification_report(y_test, RFC_pre))





# ##  XGBoost
# from xgboost import XGBRegressor

# XGB=XGBRegressor(n_estimators=80,max_depth=8,learning_rate=0.1,random_state=42) # enhance learning_rate from 0.05 to 0.1
# XGB.fit(x_train,y_train,eval_set=[(x_test,y_test)],verbose=False)
# XGB_pre = XGB.predict(x_test)
# mse_XGB = mean_squared_error(y_test, XGB_pre)
# rmse_XGB = np.sqrt(mse_XGB)
# mae_XGB = mean_absolute_error(y_test, XGB_pre)
# r2_XGB = r2_score(y_test, XGB_pre)

# print(mse_XGB,rmse_XGB,mae_XGB,r2_XGB)

# ## XGBoost - Classifier
from xgboost import XGBClassifier

XGBC=XGBClassifier(n_estimators=350, learning_rate=0.07, max_depth=6, reg_alpha=0.01, random_state=42)
XGBC.fit(x_train,y_train)
XGBC_pre=XGBC.predict(x_test)


accuracy_XGBC=accuracy_score(y_test, XGBC_pre)
print(f"\nXGBoost Accuracy: {accuracy_XGBC:.4f}")
print(classification_report(y_test, XGBC_pre))








