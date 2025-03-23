from A_clean import data_df
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# print(data_df)

# # 01. Pearson and Heatmap - continuous
# data_df_p1=data_df.drop(columns=['id','year','popularity_new'])
# data_pearson=data_df_p1.corr()
# plt.figure(figsize=(20, 8))
# sb.heatmap(data_pearson, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5, cbar=True)
# plt.title('Correlation Heatmap of Red Wine Features')
# plt.show()

# data_pearson=data_pearson.drop(columns=['valence', 'acousticness',  'danceability',
#        'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
#        'liveness', 'loudness', 'mode', 'speechiness', 'tempo'])
# data_pearson_f1=data_pearson[data_pearson["popularity"] > 0]
# print(data_pearson_f1)



# # 02. Pearson and Heatmap - binary

# data_df_p2=data_df.drop(columns=['id','year','popularity'])
# data_pearson=data_df_p2.corr()
# plt.figure(figsize=(20, 8))
# sb.heatmap(data_pearson, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5, cbar=True)
# plt.title('Correlation Heatmap of Red Wine Features')
# plt.show()

# data_pearson=data_pearson.drop(columns=['valence', 'acousticness',  'danceability',
#        'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
#        'liveness', 'loudness', 'mode', 'speechiness', 'tempo'])
# data_pearson_f2=data_pearson[data_pearson["popularity_new"] > 0]
# print(data_pearson_f2)


# ####################################################################################################################

# # 03. PCA - continuous

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# pca_features=['valence', 'acousticness',  'danceability',
#        'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
#        'liveness', 'loudness', 'mode', 'speechiness', 'tempo']

# x_pcaChosen=data_df[pca_features]
# scaler=StandardScaler()
# x_scale=scaler.fit_transform(x_pcaChosen)

# pcaTest=PCA(n_components=len(pca_features))
# x_pcaTest=pcaTest.fit_transform(x_scale)

# # Getting PCA result
# pca_components_df=pd.DataFrame(x_pcaTest, columns=[f"PC{i+1}" for i in range(len(pca_features))])
# pca_components_df["popularity"]=data_df["popularity"]

# # Calculate the importance of each feature
# correlation_with_popularity=pca_components_df.corr()["popularity"].drop("popularity").abs().sort_values(ascending=False)

# # Top 5
# top_5pca=correlation_with_popularity.index[:5]
# load_all_pca=pd.DataFrame(pcaTest.components_.T, index=pca_features, columns=[f"PC{i+1}" for i in range(len(pca_features))])
# feature_importance=load_all_pca[top_5pca].abs().sum(axis=1).sort_values(ascending=False)

# # Into 3 group
# Group01=list(feature_importance.index[:5])
# Group02=list(feature_importance.index[5:10])
# Group03=list(feature_importance.index[10:])

# Group01 += [""] * (5 - len(Group01))  
# Group02 += [""] * (5 - len(Group02))  
# Group03 += [""] * (5 - len(Group03))

# groups_features = pd.DataFrame({
#     "Group 01 (Most Important)": Group01,
#     "Group 02 (Moderate Importance)": Group02,
#     "Group 03 (Less Important)": Group03[:5]
# })

# print("\nTarget: popularity")
# print(groups_features)




# # 04. PCA - binary


# x_pcaChosen=data_df[pca_features]
# scaler=StandardScaler()
# x_scale=scaler.fit_transform(x_pcaChosen)

# pcaTest=PCA(n_components=len(pca_features))
# x_pcaTest=pcaTest.fit_transform(x_scale)

# # Getting PCA result
# pca_components_df=pd.DataFrame(x_pcaTest, columns=[f"PC{i+1}" for i in range(len(pca_features))])
# pca_components_df["popularity_new"]=data_df["popularity_new"]

# # Calculate the importance of each feature
# correlation_with_popularity=pca_components_df.corr()["popularity_new"].drop("popularity_new").abs().sort_values(ascending=False)

# # Top 5
# top_5pca=correlation_with_popularity.index[:5]
# load_all_pca=pd.DataFrame(pcaTest.components_.T, index=pca_features, columns=[f"PC{i+1}" for i in range(len(pca_features))])
# feature_importance=load_all_pca[top_5pca].abs().sum(axis=1).sort_values(ascending=False)

# # Into 3 group
# Group01=list(feature_importance.index[:5])
# Group02=list(feature_importance.index[5:10])
# Group03=list(feature_importance.index[10:])

# Group01 += [""] * (5 - len(Group01))  
# Group02 += [""] * (5 - len(Group02))  
# Group03 += [""] * (5 - len(Group03))

# groups_features = pd.DataFrame({
#     "Group 01 (Most Important)": Group01,
#     "Group 02 (Moderate Importance)": Group02,
#     "Group 03 (Less Important)": Group03[:5]
# })

# print("\nTarget: popularity_new")
# print(groups_features)








