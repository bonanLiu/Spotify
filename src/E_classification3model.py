import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.stats import zscore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

data_df=pd.read_csv("data.csv")
# print(data_df['popularity'].unique())

data_df['popularity_new'] = np.where(data_df['popularity'].between(1, 50), 0, 1)
# print(data_df['popularity_new'].value_counts())

## Reorganize the columns
data_df = data_df.drop(columns=['name', 'artists','release_date'])
data_df = data_df[[ 'id','year','valence', 'acousticness',  'danceability',
       'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'popularity','popularity_new']]

##  Basic Information
# print(data_df.info(), data_df.describe(), data_df.shape)


## Data Cleaning
# print("Data shape before cleaning:")
# print(data_df.shape)

data_df.isnull().sum()
data_df.isna().sum()
data_df=data_df.drop_duplicates()


## Find Outliers - All features (x values)
ZS=zscore( data_df[['valence', 'acousticness',  'danceability',
       'duration_ms', 'energy', 'explicit',  'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo']])
outliers=(abs(ZS>5).sum(axis=0))
outliersMark=abs(ZS)>5
outliersData=data_df[outliersMark.any(axis=1)]

data_df = data_df[(abs(ZS)<=5).all(axis=1)]

print("Data shape After cleaning:")
print(data_df.shape)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print(data_df)
# Separate features and target
# We drop "popularity" since "popularity_new" is our binary target.
X = data_df[['valence', 'explicit', 'instrumentalness', 'danceability', 'loudness',
                         'acousticness', 'liveness', 'duration_ms', 'tempo', 'energy']]
y = data_df["popularity_new"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features: scaling is especially important for KNN and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- K-Nearest Neighbors (KNN) ---
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(y_test, knn_preds))
print("KNN Classification Report:")
print(classification_report(y_test, knn_preds))
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_preds))
print("KNN Precision:", precision_score(y_test, knn_preds))
print("KNN Recall:", recall_score(y_test, knn_preds))
print("KNN F1 Score:", f1_score(y_test, knn_preds))
if hasattr(knn, 'predict_proba'):
    knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
    print("KNN ROC-AUC Score:", roc_auc_score(y_test, knn_probs))
end_time = time.time()
print("Runtime: {:.4f} seconds".format(end_time - start_time))

# --- Naïve Bayes ---
start_time = time.time()
nb = GaussianNB()  
nb.fit(X_train_scaled, y_train)
nb_preds = nb.predict(X_test_scaled)
print("Naïve Bayes Accuracy:", accuracy_score(y_test, nb_preds))
print("Naïve Bayes Classification Report:")
print(classification_report(y_test, nb_preds))
print("Naïve Bayes Confusion Matrix:")
print(confusion_matrix(y_test, nb_preds))
print("Naïve Bayes Precision:", precision_score(y_test, nb_preds))
print("Naïve Bayes Recall:", recall_score(y_test, nb_preds))
print("Naïve Bayes F1 Score:", f1_score(y_test, nb_preds))
if hasattr(nb, 'predict_proba'):
    nb_probs = nb.predict_proba(X_test_scaled)[:, 1]
    print("Naïve Bayes ROC-AUC Score:", roc_auc_score(y_test, nb_probs))
end_time = time.time()
print("Runtime: {:.4f} seconds".format(end_time - start_time))

# --- Support Vector Machine (SVM) ---
start_time = time.time()
svm = SVC(kernel='rbf', C=1, gamma='scale', probability=False) 
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))
print("SVM Precision:", precision_score(y_test, svm_preds))
print("SVM Recall:", recall_score(y_test, svm_preds))
print("SVM F1 Score:", f1_score(y_test, svm_preds))
if hasattr(svm, 'predict_proba'):
    svm_probs = svm.predict_proba(X_test_scaled)[:, 1]
    print("SVM ROC-AUC Score:", roc_auc_score(y_test, svm_probs))
end_time = time.time()
print("Runtime: {:.4f} seconds".format(end_time - start_time))

# --- Performance Summary  ---
# KNN:
# - KNN is simple and effective for small datasets.
# - Accuracy depends on the number of neighbors and scaling.
# - Moderate training time; can be slow on large datasets.

# Naïve Bayes:
# - Very fast to train and predict.
# - Performs well on normally distributed features.
# - May underperform if features are highly correlated.

# SVM:
# - Powerful with non-linear kernels like RBF.
# - Can be slow with large datasets and many features.
# - Tends to have high accuracy but longer runtime.
