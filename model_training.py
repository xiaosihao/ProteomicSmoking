###Classification
#-----------------
import optuna  # import libraries
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sqlalchemy import create_engine
import pandas as pd 
import pickle
from sklearn.base import clone
from shaphypetune import BoostBoruta
from machine_learning import *
import pickle
import gzip
import plotly.io as pio  # plot results
from sklearn.model_selection import train_test_split  # split data
from SHAP_RFECV import *


n_trials = 500
early_stopping_rounds = 20
random_state = 1996

param_dict = {}

#read csv file
olink_data = pd.read_feather('file_name')  # read data
olink_data = olink_data.set_index('eid')
print('read1')


ukb = pd.read_feather('file_name')
ukb = ukb.set_index('eid')
print('read2')

#remove overlapping columns
ukb = ukb.drop(columns = [i for i in ukb.columns if i in olink_data.columns])

df = ukb.join(olink_data,how='inner')

#remove samples with missing smoking status
df = df[~df['smoking_status'].isna()]

#remove samples that are never smokers but have tobacco exposure
df_never = df[df['smoking_status'] == 'Never']
df_never = df_never[~(df_never['tobacco_exposure_home']>0)]
df_never = df_never[~(df_never['tobacco_exposure_outside']>0)]

df_current = df[df['smoking_status']=='Current']


pro = olink_data.columns.to_list()
#keep samples that are current in smoking_status and those who are No in ever_smoked
df_train = pd.concat([df_never,df_current])[pro+['smoking_status']]
#reset smoking status datatype to category to only have 2 categories
df_train.smoking_status = df_train.smoking_status.astype('str')
df_train.smoking_status = df_train.smoking_status.astype('category')

# Convert Current to 1 and Never to 0 in df_train
df_train['smoking_status'] = df_train['smoking_status'].replace({'Current':1,'Never':0})

train_df, test_df = train_test_split(df_train, test_size=0.3,random_state=random_state,stratify=df_train['smoking_status'])
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=random_state,stratify=train_df['smoking_status'])

X_train = pd.concat([train_df.drop(['smoking_status'],1),val_df.drop(['smoking_status'],1)])
y_train = pd.concat([train_df['smoking_status'],val_df['smoking_status']])


def optuna_lgbm(X, y,storage,study_name,n_trials,early_stopping_rounds):
    # make sqlite database engine to run with optuna
    engine = create_engine(storage, echo=False)

    def objective(trial):
        params = {
            'objective': 'binary',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 5000,
            'num_leaves': trial.suggest_int('num_leaves', 2, 100),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 100),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1),
            'random_state': random_state,
            'metric': 'None',
            'n_jobs': -1
        }
        
        #Stratified KFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        auc_scores = []
        for train_idx, val_idx in cv.split(X,y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
        
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric=['auc'], early_stopping_rounds=early_stopping_rounds, verbose=False)  # fit model
        
            y_pred = model.predict_proba(X_val_fold)[:, 1]  # predict
            auc_score = roc_auc_score(y_val_fold, y_pred)

            auc_scores.append(auc_score)
        
        return np.mean(auc_scores)

    # Run the optimization using optuna
    study = optuna.create_study(direction='maximize',storage=storage,study_name=study_name)
    study.optimize(objective, n_trials=n_trials)

    return study


# Run optuna
storage = f'sqlite:///file_name'

study1 = optuna_lgbm(X_train, y_train,storage,'init',n_trials,early_stopping_rounds)

# Get the best hyperparameters and train the final model
best_params = study1.best_params

base_params = {'boosting_type': 'gbdt','metric':"None",'n_estimators':5000,'n_jobs':-1,'random_state':random_state}
base_params.update(best_params)
best_model =lgb.LGBMClassifier(**base_params)

#save best model
param_dict['init'] = best_model


with open('file_name', 'wb') as f:
    pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print('finish')


#Boruta
X_train = train_df.drop(['smoking_status'],1)
y_train = train_df['smoking_status']

X_val = val_df.drop(['smoking_status'],1)
y_val = val_df['smoking_status']

clf = clone(param_dict['init'])

model = BoostBoruta(clf, max_iter=200, perc=100,n_jobs=-1,importance_type='shap_importances')
model.fit(X_train,y_train, eval_set=[(X_val,y_val)], early_stopping_rounds=20,eval_metric='auc',verbose=False)

with open(f'file_name', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
print('finish')

#Tune Boruta model
X_train = pd.concat([train_df.drop(['smoking_status'],1),val_df.drop(['smoking_status'],1)]).iloc[:,model.support_]
y_train = pd.concat([train_df['smoking_status'],val_df['smoking_status']])

study2 = optuna_lgbm(X_train, y_train,storage,'boruta',n_trials,early_stopping_rounds)

# Get the best hyperparameters and train the final model
best_params = study2.best_params

base_params = {'boosting_type': 'gbdt','metric':"None",'n_estimators':5000,'n_jobs':-1,'random_state':random_state}
base_params.update(best_params)
best_model =lgb.LGBMClassifier(**base_params)

#save best model
param_dict['boruta'] = best_model


with open('file_name', 'wb') as f:
    pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print('finish')