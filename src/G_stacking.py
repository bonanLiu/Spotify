import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Start timing
t_start = time.time()

# Load and preprocess data
data_df = pd.read_csv("data.csv")
data_df['popularity_new'] = np.where(data_df['popularity'].between(1, 50), 0, 1)
data_df = data_df.drop(columns=['name', 'artists','release_date'])
data_df = data_df.drop_duplicates()

# Feature engineering
data_df['mood'] = data_df['valence'] * data_df['energy']
data_df['rhythm'] = data_df['tempo'] / data_df['duration_ms']
data_df['duration_log'] = np.log1p(data_df['duration_ms'])
data_df['tempo_log'] = np.log1p(data_df['tempo'])

features = ['valence', 'explicit', 'instrumentalness', 'danceability', 'loudness',
            'acousticness', 'liveness', 'energy', 'mood', 'rhythm', 'duration_log', 'tempo_log']
X = data_df[features]
y = data_df['popularity_new']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning for LGBM
lgbm_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.03, 0.05, 0.07, 0.1, 0.2],
    'max_depth': [3, 4, 6, 8, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

lgbm_base = LGBMClassifier(random_state=42)
lgbm_search = RandomizedSearchCV(lgbm_base, lgbm_params, n_iter=10, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
lgbm_search.fit(X_train_res, y_train_res)
lgbm_best = lgbm_search.best_estimator_

# CatBoost
cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42)

# XGBoost
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric='logloss', random_state=42)

# Stacking classifier
stack_model = StackingClassifier(
    estimators=[
        ('lgbm', lgbm_best),
        ('cat', cat),
        ('xgb', xgb),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
    ],
    final_estimator=RidgeClassifier(),
    n_jobs=-1
)

# Train stacking model
stack_model.fit(X_train_res, y_train_res)

# Evaluate
preds = stack_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1 Score:", f1_score(y_test, preds))

# ROC-AUC (if model supports predict_proba)
if hasattr(stack_model, 'predict_proba'):
    probs = stack_model.predict_proba(X_test_scaled)[:, 1]
    print("ROC-AUC Score:", roc_auc_score(y_test, probs))

# End timing and print runtime (excluding SHAP)
t_end = time.time()
print(f"[Model Training + Prediction] Runtime: {t_end - t_start:.2f} seconds")



## OUTPUT
# Accuracy: 0.7363100993231959
#               precision    recall  f1-score   support

#            0       0.77      0.82      0.80     21295
#            1       0.67      0.59      0.63     12836

#     accuracy                           0.74     34131
#    macro avg       0.72      0.71      0.71     34131
# weighted avg       0.73      0.74      0.73     34131

# [Model Training + Prediction] Runtime: 490.87 seconds