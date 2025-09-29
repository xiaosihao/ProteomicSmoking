##calculating contribution
#-------------------------
import pandas as pd  # import libraries
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score,precision_score
from sklearn.model_selection import train_test_split  # split data
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
from machine_learning import *
import pickle
from sklearn.base import clone
from sqlalchemy import create_engine
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from shaphypetune import BoostBoruta
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

param_dict = {}

n_trials = 150
early_stopping_rounds = 20
random_state = 1996

#read genotype data
genotype = pd.read_feather('file_name')  # read data
genotype.set_index('eid', inplace=True)
snp_list = pd.read_csv('file_name',sep='\t')
snp_list= snp_list['rsID'].to_list()
snp_list = [x for x in snp_list if x in genotype.columns]
genotype = genotype[snp_list]

#read xwas data
xwas_coded = pd.read_feather('file_name')
xwas_coded.set_index('eid', inplace=True)
#read behavioural data
behavioural = pd.read_feather('file_name')
behavioural.set_index('eid', inplace=True)

#merge all data
xwas = xwas_coded.join(genotype, how='left')
xwas = xwas.join(behavioural, how='left')

#read pSIN
all_preds_df = pd.read_csv('file_name', index_col=0)
xwas = xwas.join(all_preds_df['y_pred'], how='left')


train_df, test_df = train_test_split(xwas, test_size=0.3,random_state=1996,shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=1996,shuffle=True)

X_train = pd.concat([train_df.drop(['y_pred'],1),val_df.drop(['y_pred'],1)]) 
y_train = pd.concat([train_df['y_pred'],val_df['y_pred']])

def optuna_lgbm(X, y,storage,study_name,n_trials,early_stopping_rounds):
    # make sqlite database engine to run with optuna
    engine = create_engine(storage, echo=False)

    def objective(trial):
        params = {
            'objective': 'regression',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 5000,
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, y.shape[0]*0.8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1,log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100,log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1,log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1,log=True),
            'random_state': random_state,
            'metric': 'None',
            'n_jobs': -1
        }
        
        #Stratified KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_scores = []
        for train_idx, val_idx in cv.split(X,y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
        
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)  # fit model
        
            y_pred = model.predict(X_val_fold)  # predict
            r2 = r2_score(y_val_fold, y_pred)

            r2_scores.append(r2)
        
        return np.mean(r2_scores)

    # Run the optimization using optuna
    study = optuna.create_study(direction='maximize',storage=storage,study_name=study_name,sampler=optuna.samplers.TPESampler(seed=random_state),load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)

    return study

# Run optuna
storage = f'sqlite:///file_name'

study1 = optuna_lgbm(X_train, y_train,storage,'init',n_trials,early_stopping_rounds)

# Get the best hyperparameters and train the final model
best_params = study1.best_params

base_params = {'boosting_type': 'gbdt','metric':"None",'n_estimators':5000,'n_jobs':-1,'random_state':random_state}
base_params.update(best_params)
best_model =lgb.LGBMRegressor(**base_params)

#save best model
param_dict['init'] = best_model

with open(f'file_name', 'wb') as f:
    pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print('finish')

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score,precision_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
from machine_learning import *
import pickle
from sklearn.base import clone
from sqlalchemy import create_engine
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from shaphypetune import BoostBoruta
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def optuna_lgbm(X, y,storage,study_name,n_trials,early_stopping_rounds):
    # make sqlite database engine to run with optuna
    engine = create_engine(storage, echo=False)

    def objective(trial):
        params = {
            'objective': 'regression',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 5000,
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, y.shape[0]*0.8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1,log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100,log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1,log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1,log=True),
            'random_state': random_state,
            'metric': 'None',
            'n_jobs': -1
        }
        
        #Stratified KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_scores = []
        for train_idx, val_idx in cv.split(X,y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
        
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
        
            y_pred = model.predict(X_val_fold)
            r2 = r2_score(y_val_fold, y_pred)

            r2_scores.append(r2)
        
        return np.mean(r2_scores)

    # Run the optimization using optuna
    study = optuna.create_study(direction='maximize',storage=storage,study_name=study_name,sampler=optuna.samplers.TPESampler(seed=random_state),load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)

    return study

param_dict = {}

n_trials = 150
early_stopping_rounds = 20
random_state = 1996
tag_list = ['genotype','exposome','behaviour']
data_dict = {}

#read genotype data
genotype = pd.read_feather('file_name')
genotype.set_index('eid', inplace=True)
snp_list = pd.read_csv('file_name',sep='\t')
snp_list= snp_list['rsID'].to_list()
snp_list = [x for x in snp_list if x in genotype.columns]
genotype = genotype[snp_list]

data_dict['genotype'] = genotype

#read xwas data
xwas_coded = pd.read_feather('file_name')
xwas_coded.set_index('eid', inplace=True)

data_dict['exposome'] = xwas_coded

#read behavioural data
behavioural = pd.read_feather('file_name')
behavioural.set_index('eid', inplace=True)

data_dict['behaviour'] = behavioural


#read pSIN
all_preds_df = pd.read_csv('file_name', index_col=0)

for tag in tag_list:
    data_dict[tag] = all_preds_df[['y_pred']].join(data_dict[tag], how='left')

    train_df, test_df = train_test_split(data_dict[tag], test_size=0.3,random_state=1996,shuffle=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=1996,shuffle=True)

    X_train = pd.concat([train_df.drop(['y_pred'],1),val_df.drop(['y_pred'],1)]) 
    y_train = pd.concat([train_df['y_pred'],val_df['y_pred']])

    # Run optuna
    storage = f'sqlite:///file_name'

    study1 = optuna_lgbm(X_train, y_train,storage,tag,n_trials,early_stopping_rounds)

    # Get the best hyperparameters and train the final model
    best_params = study1.best_params

    base_params = {'boosting_type': 'gbdt','metric':"None",'n_estimators':5000,'n_jobs':-1,'random_state':random_state}
    base_params.update(best_params)
    best_model =lgb.LGBMRegressor(**base_params)

    #save best model
    param_dict[tag] = best_model

with open(f'../../data/pSIN/param_dict_pSIN_all_split.p', 'wb') as f:
    pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print('finish')
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score,precision_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
from machine_learning import *
import pickle
from sklearn.base import clone
from sqlalchemy import create_engine
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from shaphypetune import BoostBoruta
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def optuna_lgbm(X, y,storage,study_name,n_trials,early_stopping_rounds):
    # make sqlite database engine to run with optuna
    engine = create_engine(storage, echo=False)

    def objective(trial):
        params = {
            'objective': 'regression',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 5000,
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, y.shape[0]*0.8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1,log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100,log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1,log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1,log=True),
            'random_state': random_state,
            'metric': 'None',
            'n_jobs': -1
        }
        
        #Stratified KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_scores = []
        for train_idx, val_idx in cv.split(X,y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
        
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
        
            y_pred = model.predict(X_val_fold)
            r2 = r2_score(y_val_fold, y_pred)

            r2_scores.append(r2)
        
        return np.mean(r2_scores)

    # Run the optimization using optuna
    study = optuna.create_study(direction='maximize',storage=storage,study_name=study_name,sampler=optuna.samplers.TPESampler(seed=random_state),load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)

    return study

param_dict = {}

n_trials = 150
early_stopping_rounds = 20
random_state = 1996
tag_list = ['genotype','exposome','behaviour']
data_dict = {}

#read genotype data
genotype = pd.read_feather('file_name')
genotype.set_index('eid', inplace=True)
snp_list = pd.read_csv('file_name',sep='\t')
snp_list= snp_list['rsID'].to_list()
snp_list = [x for x in snp_list if x in genotype.columns]
genotype = genotype[snp_list]

#read behavioural data
behavioural = pd.read_feather('file_name')
behavioural.set_index('eid', inplace=True)

#merge genotype and behavioural data
data = genotype.join(behavioural, how='inner')

#read pSIN
all_preds_df = pd.read_csv('file_name', index_col=0)


data = all_preds_df[['y_pred']].join(data, how='left')

train_df, test_df = train_test_split(data, test_size=0.3,random_state=1996,shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=1996,shuffle=True)

X_train = pd.concat([train_df.drop(['y_pred'],1),val_df.drop(['y_pred'],1)]) 
y_train = pd.concat([train_df['y_pred'],val_df['y_pred']])

# Run optuna
storage = f'sqlite:///file_name'

study1 = optuna_lgbm(X_train, y_train,storage,'init',n_trials,early_stopping_rounds)

# Get the best hyperparameters and train the final model
best_params = study1.best_params

base_params = {'boosting_type': 'gbdt','metric':"None",'n_estimators':5000,'n_jobs':-1,'random_state':random_state}
base_params.update(best_params)
best_model =lgb.LGBMRegressor(**base_params)

#save best model
param_dict['all'] = best_model

with open(f'file_name', 'wb') as f:
    pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print('finish')

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score,precision_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
from machine_learning import *
import pickle
from sklearn.base import clone
from sqlalchemy import create_engine
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from shaphypetune import BoostBoruta
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def optuna_lgbm(X, y,storage,study_name,n_trials,early_stopping_rounds):
    # make sqlite database engine to run with optuna
    engine = create_engine(storage, echo=False)

    def objective(trial):
        params = {
            'objective': 'regression',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'n_estimators': 5000,
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, y.shape[0]*0.8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1,log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100,log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1,log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1,log=True),
            'random_state': random_state,
            'metric': 'None',
            'n_jobs': -1
        }
        
        #Stratified KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_scores = []
        for train_idx, val_idx in cv.split(X,y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
        
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
        
            y_pred = model.predict(X_val_fold)
            r2 = r2_score(y_val_fold, y_pred)

            r2_scores.append(r2)
        
        return np.mean(r2_scores)

    # Run the optimization using optuna
    study = optuna.create_study(direction='maximize',storage=storage,study_name=study_name,sampler=optuna.samplers.TPESampler(seed=random_state),load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)

    return study

param_dict = {}

n_trials = 150
early_stopping_rounds = 20
random_state = 1996
tag_list = ['genotype','exposome','behaviour']
data_dict = {}

#read genotype data
genotype = pd.read_feather('file_name')
genotype.set_index('eid', inplace=True)
snp_list = pd.read_csv('file_name',sep='\t')
snp_list= snp_list['rsID'].to_list()
snp_list = [x for x in snp_list if x in genotype.columns]
genotype = genotype[snp_list]

#read behavioural data
behavioural = pd.read_feather('file_name')
behavioural.set_index('eid', inplace=True)

#read xwas data
xwas_coded = pd.read_feather('file_name')
xwas_coded.set_index('eid', inplace=True)

#read clinical data
clinical = pd.read_feather('file_name')
clinical.set_index('eid', inplace=True)

#merge all data
data = xwas_coded.join(genotype, how='left')
data = data.join(behavioural, how='left')
data = data.join(clinical, how='left')


#read pSIN
all_preds_df = pd.read_csv('file_name', index_col=0)


data = all_preds_df[['y_pred']].join(data, how='left')

train_df, test_df = train_test_split(data, test_size=0.3,random_state=1996,shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=1996,shuffle=True)

X_train = pd.concat([train_df.drop(['y_pred'],1),val_df.drop(['y_pred'],1)]) 
y_train = pd.concat([train_df['y_pred'],val_df['y_pred']])

# Run optuna
storage = f'sqlite:///file_name'

study1 = optuna_lgbm(X_train, y_train,storage,'init',n_trials,early_stopping_rounds)

# Get the best hyperparameters and train the final model
best_params = study1.best_params

base_params = {'boosting_type': 'gbdt','metric':"None",'n_estimators':5000,'n_jobs':-1,'random_state':random_state}
base_params.update(best_params)
best_model =lgb.LGBMRegressor(**base_params)

#save best model
param_dict['all'] = best_model

with open(f'file_name', 'wb') as f:
    pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print('finish')

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score,precision_score,explained_variance_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
from machine_learning import *
import shap
import pickle
from sklearn.model_selection import KFold
from sklearn.utils import resample
from tqdm import tqdm

random_state = 1996
early_stopping_rounds = 20

param_dict = {}
for tag in ['all','current','previous','never']:
    param_dict[tag] = {}

    with open(f'file_name', 'rb') as handle:
        temp = pickle.load(handle)
    param_dict[tag]['G'] = temp['genotype']

    with open(f'file_name', 'rb') as handle:
        temp = pickle.load(handle)
    param_dict[tag]['G+S'] = temp[tag]

    with open(f'file_name', 'rb') as handle:
        temp = pickle.load(handle)
    param_dict[tag]['G+S+E'] = temp['init']

    with open(f'file_name', 'rb') as handle:
        temp = pickle.load(handle)
    param_dict[tag]['G+S+E+C'] = temp[tag]

    data_raw_dict = {}
#read genotype data
genotype = pd.read_feather('file_name')
genotype.set_index('eid', inplace=True)
snp_list = pd.read_csv('file_name',sep='\t')
snp_list= snp_list['rsID'].to_list()
snp_list = [x for x in snp_list if x in genotype.columns]
genotype = genotype[snp_list]
data_raw_dict['genotype'] = genotype

#read xwas data
xwas_coded = pd.read_feather('file_name')
xwas_coded.set_index('eid', inplace=True)
data_raw_dict['exposome'] = xwas_coded

#read behavioural data
behavioural = pd.read_feather('file_name')
behavioural.set_index('eid', inplace=True)
data_raw_dict['behaviour'] = behavioural

#read clinical data
clinical = pd.read_feather('file_name')
clinical.set_index('eid', inplace=True)
data_raw_dict['clinical'] = clinical

#read pSIN
all_preds_df = pd.read_csv('file_name', index_col=0)

data_dict = {}
for tag, status in zip(['all','current','previous','never'], ['All','Current','Previous','Never']):
    data_dict[tag] = {}
    for cat in ['G','G+S','G+S+E','G+S+E+C']:
        preds_df_temp = all_preds_df.copy()
        if tag in ['current','previous','never']:
            preds_df_temp = preds_df_temp[preds_df_temp['smoking_status']==status]

        if cat == 'G':
            temp_df = genotype.copy()
        elif cat == 'G+S':
            temp_df = genotype.join(behavioural, how='left')
        elif cat == 'G+S+E':
            temp_df = genotype.join(behavioural, how='left').join(xwas_coded, how='left')
        elif cat == 'G+S+E+C':
            temp_df = genotype.join(behavioural, how='left').join(xwas_coded, how='left').join(clinical, how='left')

        data_dict[tag][cat] = preds_df_temp[['y_pred']].join(temp_df, how='left')
        #remove columns starting with 'smoking_status'
        if tag in ['current','previous','never']:
            data_dict[tag][cat] = data_dict[tag][cat][data_dict[tag][cat].columns.drop(list(data_dict[tag][cat].filter(regex='smoking_status')))]
        #remove columns with all 0 or NA
        data_dict[tag][cat] = data_dict[tag][cat].loc[:, ~((data_dict[tag][cat] == 0).all(axis=0) | data_dict[tag][cat].isna().all(axis=0))]
        #remove columns with all 0 or NA
        data_dict[tag][cat] = data_dict[tag][cat].loc[:, ~((data_dict[tag][cat] == 1).all(axis=0) | data_dict[tag][cat].isna().all(axis=0))]

ev_test = {}

for tag in tqdm(['all','current','previous','never']):
    ev_test[tag] = {}
    for cat in tqdm(['G','G+S','G+S+E','G+S+E+C']):

        train_df, test_df = train_test_split(data_dict[tag][cat], test_size=0.3, random_state=random_state)
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=random_state)

        #test set
        X_train = train_df.drop(['y_pred'],axis=1)
        y_train = train_df['y_pred']

        X_val = val_df.drop(['y_pred'],axis=1)
        y_val = val_df['y_pred']

        X_test = test_df.drop(['y_pred'],axis=1)
        y_test = test_df['y_pred']

        #model
        model = clone(param_dict[tag][cat])
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=0)

        y_pred = model.predict(X_test)

        ev_test[tag][cat] = explained_variance_score(y_test, y_pred)

#calculate the difference
ev_diff = {}
for tag in ['all','current','previous','never']:
    ev_diff[tag] = {}
    ev_diff[tag]['G'] = ev_test[tag]['G']

    ev_diff[tag]['S'] = ev_test[tag]['G+S'] - ev_test[tag]['G']

    ev_diff[tag]['E'] = ev_test[tag]['G+S+E'] - ev_test[tag]['G+S']

    ev_diff[tag]['C'] = ev_test[tag]['G+S+E+C'] - ev_test[tag]['G+S+E']
 
 import matplotlib.pyplot as plt  # plot results
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from adjustText import adjust_text
import matplotlib.gridspec as gridspec

# Prepare the data for pie chart visualization
formatted_data = {}
for key, values in ev_diff.items():
    total_explained = sum(values.values())
    not_explained = max(1 - total_explained, 0)  # Ensure non-negative

    # Reorder and update keys for new labels in the desired sequence
    new_values = {'Genome': values.get('G', 0),
                  'Smoking-related variables': values.get('S', 0),
                  'Social-demographic and lifestyles': values.get('E', 0),
                  'Clinical biomarkers and risk factors': values.get('C', 0),
                  'Not explained': not_explained}
    formatted_data[key] = new_values

# Normalize so that the sum is 1
for key, values in formatted_data.items():
    total_explained = sum(values.values())
    for k, v in values.items():
        formatted_data[key][k] = v / total_explained

df = pd.DataFrame(formatted_data).T
df = df[['Genome', 'Smoking-related variables', 'Social-demographic and lifestyles', 'Clinical biomarkers and risk factors']] * 100

fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
ax = plt.subplot(gs[0, 0], aspect='auto')  # Span the entire left column

color_list = ['#b3e2cd', '#cbd5e8', '#fdcdac', '#ffff99', '#cccccc']

# Plot a horizontal stacked bar plot
plot = df.plot(kind='barh', stacked=True, ax=ax, color=color_list, width=0.8)

# Title and labels
plt.xlabel('Variance explained (%)', fontsize=16)
plt.yticks(rotation=0, fontsize=14)
ax.set_yticklabels(['Whole population', 'Current smokers', 'Previous smokers', 'Never smokers'])

ax.grid(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Invert the y-axis to reverse the order
ax.invert_yaxis()

# Adding and adjusting text annotations
texts = []

for i, bar in enumerate(plot.containers):
    texts += ax.bar_label(bar, label_type='center', color='black', fontsize=12, padding=5, fmt='%.1f%%')
texts[3].set_position((-1, 0))
texts[7].set_position((15, 0))
texts[11].set_position((25, 0))
texts[12].set_position((20, 0))
texts[13].set_position((10, 0))
texts[14].set_position((10, 0))
texts[15].set_position((20, 0))
# Add a legend with the new order
patches = [mpatches.Patch(color=col, label=lab) for col, lab in zip(color_list, ['Genome', 'Smoking history', 'Social-demographic\n and lifestyles', 'Clinical biomarkers and risk factors'])]
plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12)

plt.show()