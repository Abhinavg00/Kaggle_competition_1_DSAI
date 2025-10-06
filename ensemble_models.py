import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

train_df = pd.read_csv('train.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Missing values in training data: {train_df.isnull().sum().sum()}")

X = train_df.drop(columns=['id', 'song_popularity'])
y = train_df['song_popularity']


print(f"Class balance: {y.value_counts(normalize=True).round(4)}")

# Below, some feature engineering is applied as suggested by deepseek r1

if 'key' in X.columns:
    print("   Applying circular encoding to 'key' feature...")
    X['key_sin'] = np.sin(2 * np.pi * X['key'] / 12.0)
    X['key_cos'] = np.cos(2 * np.pi * X['key'] / 12.0)
    X = X.drop(columns=['key'])


if 'energy' in X.columns and 'danceability' in X.columns:
    X['energy_danceability'] = X['energy'] * X['danceability']
if 'valence' in X.columns and 'energy' in X.columns:
    X['valence_energy'] = X['valence'] * X['energy']
if 'acousticness' in X.columns and 'instrumentalness' in X.columns:
    X['acoustic_instrumental'] = X['acousticness'] * X['instrumentalness']
if 'tempo' in X.columns and 'energy' in X.columns:
    X['tempo_energy_ratio'] = X['tempo'] / (X['energy'] + 1e-8)
if 'loudness' in X.columns and 'energy' in X.columns:
    X['loudness_energy_interaction'] = X['loudness'] * X['energy']



key_features = ['danceability', 'energy', 'valence', 'tempo']
for feature in key_features:
    if feature in X.columns:
        X[f'{feature}_squared'] = X[feature] ** 2


print(f"   Shape after feature engineering: {X.shape}")
feature_names = X.columns.tolist()

# Feature engineering done.

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")


# using k-fold cross-validation
imputer = KNNImputer(n_neighbors=5)
# Fit on the training data only
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)
X_val = pd.DataFrame(imputer.transform(X_val), columns=feature_names)


# Feature Scaling
print("   Scaling features...")
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_names)



study_results = {}
N_SPLITS = 5 # Using 5 folds for cross-validation

# The data used for Optuna is the training set, which will be split into folds internally.
# We will use X_val later to get a final score before retraining on all data.
X_opt, y_opt = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])



# we will create separate functions for each model to optimize
# we have catboost, xgboost, lightgbm and ensemble of the three
def optimize_catboost(trial):
    params = {
        'objective': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0, 'random_seed': 42,
        'iterations': trial.suggest_int('iterations', 800, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
    }
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X_opt, y_opt):
        X_train_fold, X_val_fold = X_opt.iloc[train_idx], X_opt.iloc[val_idx]
        y_train_fold, y_val_fold = y_opt.iloc[train_idx], y_opt.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=50, verbose=0)
        preds = model.predict_proba(X_val_fold)[:, 1]
        aucs.append(roc_auc_score(y_val_fold, preds))
        
    return np.mean(aucs)


def optimize_xgboost(trial):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'early_stopping_rounds': 100, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
        'n_estimators': trial.suggest_int('n_estimators', 800, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X_opt, y_opt):
        X_train_fold, X_val_fold = X_opt.iloc[train_idx], X_opt.iloc[val_idx]
        y_train_fold, y_val_fold = y_opt.iloc[train_idx], y_opt.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        preds = model.predict_proba(X_val_fold)[:, 1]
        aucs.append(roc_auc_score(y_val_fold, preds))
        
    return np.mean(aucs)


def optimize_lightgbm(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1,
        'n_estimators': trial.suggest_int('n_estimators', 800, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X_opt, y_opt):
        X_train_fold, X_val_fold = X_opt.iloc[train_idx], X_opt.iloc[val_idx]
        y_train_fold, y_val_fold = y_opt.iloc[train_idx], y_opt.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict_proba(X_val_fold)[:, 1]
        aucs.append(roc_auc_score(y_val_fold, preds))
        
    return np.mean(aucs)


models_to_optimize = [
    ('CatBoost', optimize_catboost),
    ('XGBoost', optimize_xgboost),
    ('LightGBM', optimize_lightgbm)
]

for model_name, objective_func in models_to_optimize:
    print(f"\n   Optimizing {model_name}...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_func, n_trials=30, timeout=1800) # n_trials can be increased for better results
    study_results[model_name] = study
    print(f"   Best {model_name} CV AUC: {study.best_value:.6f}")

# creating ensemble of the three models based on their validation AUC scores
# as ensemble was suggested in the overview of the kaggle competition
final_models = {}
final_scores = {}
val_predictions = {}

for model_name, study in study_results.items():
    best_params = study.best_params.copy()
    
    if model_name == 'CatBoost':
        model = CatBoostClassifier(**best_params, objective='Logloss', eval_metric='AUC', random_seed=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=0)
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(**best_params, objective='binary:logistic', eval_metric='auc', early_stopping_rounds=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_name == 'LightGBM':
        model = lgb.LGBMClassifier(**best_params, objective='binary', metric='auc', random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    final_models[model_name] = model
    final_scores[model_name] = auc
    val_predictions[model_name] = preds
    print(f"   Final {model_name} Validation AUC: {auc:.6f}")

# Weighted Ensemble
total_weight = sum(score ** 2 for score in final_scores.values())
weights = {name: (score ** 2) / total_weight for name, score in final_scores.items()}
ensemble_preds = sum(val_predictions[name] * w for name, w in weights.items())
ensemble_auc = roc_auc_score(y_val, ensemble_preds)
print(f"   Ensemble Validation AUC: {ensemble_auc:.6f}")



all_scores = {**final_scores, 'Ensemble': ensemble_auc}
best_model_name = max(all_scores, key=all_scores.get)
print(f"\nBest performing model: {best_model_name} (AUC: {all_scores[best_model_name]:.6f})")

# Using full dataset for final training
X_full_train_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)
X_full_train_scaled = pd.DataFrame(scaler.fit_transform(X_full_train_imputed), columns=feature_names)

final_model = None

if best_model_name != 'Ensemble':
    print(f"   Retraining {best_model_name} on the full training dataset...")
    
    best_params = study_results[best_model_name].best_params.copy()

    if best_model_name == 'CatBoost':
        final_model = CatBoostClassifier(**best_params, objective='Logloss', eval_metric='AUC', random_seed=42, verbose=0)
    elif best_model_name == 'XGBoost':
        final_model = xgb.XGBClassifier(**best_params, objective='binary:logistic', eval_metric='auc', random_state=42, n_jobs=-1)
    elif best_model_name == 'LightGBM':
        final_model = lgb.LGBMClassifier(**best_params, objective='binary', metric='auc', random_state=42, n_jobs=-1)
            
    final_model.fit(X_full_train_scaled, y)

else:
    print("   Retraining all models for the ensemble on the full training dataset...")
    for name in final_models.keys():
        best_params = study_results[name].best_params.copy()
        model_for_retraining = None
        if name == 'CatBoost':
            model_for_retraining = CatBoostClassifier(**best_params, objective='Logloss', eval_metric='AUC', random_seed=42, verbose=0)
        elif name == 'XGBoost':
            model_for_retraining = xgb.XGBClassifier(**best_params, objective='binary:logistic', eval_metric='auc', random_state=42, n_jobs=-1)
        elif name == 'LightGBM':
            model_for_retraining = lgb.LGBMClassifier(**best_params, objective='binary', metric='auc', random_state=42, n_jobs=-1)
        
        model_for_retraining.fit(X_full_train_scaled, y)
        final_models[name] = model_for_retraining

try:
    test_df = pd.read_csv('test.csv')
    test_ids = test_df['id'].copy()
    X_test = test_df.drop(columns=['id'])
except (FileNotFoundError, KeyError) as e:
    print(f"Error processing test.csv: {e}")
    exit()

# Apply same feature engineering from training
if 'key' in X_test.columns:
    X_test['key_sin'] = np.sin(2 * np.pi * X_test['key'] / 12.0)
    X_test['key_cos'] = np.cos(2 * np.pi * X_test['key'] / 12.0)
    X_test = X_test.drop(columns=['key'])
if 'energy' in X_test.columns and 'danceability' in X_test.columns: X_test['energy_danceability'] = X_test['energy'] * X_test['danceability']
if 'valence' in X_test.columns and 'energy' in X_test.columns: X_test['valence_energy'] = X_test['valence'] * X_test['energy']
if 'acousticness' in X_test.columns and 'instrumentalness' in X_test.columns: X_test['acoustic_instrumental'] = X_test['acousticness'] * X_test['instrumentalness']
if 'tempo' in X_test.columns and 'energy' in X_test.columns: X_test['tempo_energy_ratio'] = X_test['tempo'] / (X_test['energy'] + 1e-8)
if 'loudness' in X_test.columns and 'energy' in X_test.columns: X_test['loudness_energy_interaction'] = X_test['loudness'] * X_test['energy']
for feature in key_features:
    if feature in X_test.columns: X_test[f'{feature}_squared'] = X_test[feature] ** 2

X_test = X_test.reindex(columns=feature_names, fill_value=0)

# doing on the preprocessing pipeline
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=feature_names)


print("Generating final predictions")
if best_model_name == 'Ensemble':
    test_preds = sum(final_models[name].predict_proba(X_test_scaled)[:, 1] * w for name, w in weights.items())
else:
    test_preds = final_model.predict_proba(X_test_scaled)[:, 1]


submission_df = pd.DataFrame({'id': test_ids, 'song_popularity': test_preds})
submission_df.to_csv('submission.csv', index=False)

# to maximize AUC, rather than just having 0 or 1 predictions, we output the probabilities directly.

print("Process completed and file created for submission.")

