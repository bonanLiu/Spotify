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




##  Descriptive Analysis

# Pearson and Heatmap
data_df_p=data_df.drop(columns=['id'])
data_pearson=data_df_p.corr()
plt.figure(figsize=(10, 8))