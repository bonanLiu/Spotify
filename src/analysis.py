import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from scipy.stats import zscore


data_df=pd.read_csv("data.csv")


## Reorganize the columns
data_df = data_df.drop(columns=['name', 'artists','release_date'])
data_df = data_df[[ 'id','year','valence', 'acousticness',  'danceability',
       'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'popularity']]

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
print(data_df.shape)




##  Finding features

# 01. Pearson and Heatmap
data_df_p=data_df.drop(columns=['id','year'])
data_pearson=data_df_p.corr()
plt.figure(figsize=(20, 8))
sb.heatmap(data_pearson, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap of Red Wine Features')
plt.show()

data_pearson=data_pearson.drop(columns=['valence', 'acousticness',  'danceability',
       'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo'])
data_pearson_f=data_pearson[data_pearson["popularity"] > 0]
print(data_pearson_f)

# the features selected by pearson method 
#               popularity
# valence         0.011281
# danceability    0.229891
# duration_ms     0.092632
# energy          0.470917
# explicit        0.266759
# key             0.011194
# loudness        0.440592
# tempo           0.129530
# popularity      1.000000

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
pca_components_df["popularity"]=data_df["popularity"]

# Calculate the importance of each feature
correlation_with_popularity=pca_components_df.corr()["popularity"].drop("popularity").abs().sort_values(ascending=False)

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

print(groups_features)

# the features selected by PCA method 
#   Group 01 (Most Important) Group 02 (Moderate Importance) Group 03 (Less Important)
# 0                   valence                   acousticness               speechiness
# 1                  explicit                       liveness                      mode
# 2          instrumentalness                    duration_ms                       key
# 3              danceability                          tempo
# 4                  loudness                         energy


## Conclusion: Compare with the result in Pearson and PCA, we choose the 6 features to do the prediction

x=data_df['valence','explicit','danceability','loudness','tempo','energy']
y=data_df['popularity ']