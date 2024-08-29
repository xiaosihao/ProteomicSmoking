###Classification
#-----------------
import optuna
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
import plotly.io as pio
from sklearn.model_selection import train_test_split
from SHAP_RFECV import *


n_trials = 500
early_stopping_rounds = 20
random_state = 1996

param_dict = {}

#read csv file
olink_data = pd.read_feather('file_name')
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
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, y.value_counts()[1]*0.8),
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
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        auc_scores = []
        for train_idx, val_idx in cv.split(X,y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
        
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric=['auc'], early_stopping_rounds=early_stopping_rounds, verbose=False)
        
            y_pred = model.predict_proba(X_val_fold)[:, 1]
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

#Calculate pSIN and SHAP
#------------------------
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score,precision_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import numpy as np
from machine_learning import *
import pickle
random_state = 1996

#read csv file
olink_data = pd.read_feather('file_name')
olink_data = olink_data.set_index('eid')

# ukb = pd.read_feather('/Users/xiao/Library/CloudStorage/OneDrive-Nexus365/DPhil/Projects/Smoking_score/01test/data/ukb_imputation1_jul_25_2023.feather')
ukb = pd.read_feather('file_name')
ukb = ukb.set_index('eid')

#remove overlapping columns
ukb = ukb.drop(columns = [i for i in ukb.columns if i in olink_data.columns])

df = ukb.join(olink_data,how='inner')

#remove samples with missing smoking status
df = df[~df['smoking_status'].isna()]
df_never = df[df['smoking_status'] == 'Never']
df_never = df[df['smoking_status'] == 'Never']
df_never = df_never[~(df_never['tobacco_exposure_home']>0)]
df_never = df_never[~(df_never['tobacco_exposure_outside']>0)]
# df_never = df_never[~(df_never['hshld_smokers'].isin(['Yes, one household member smokes','Yes, more than one household member smokes']))]

#df_neverH is collection of df[df['smoking_status'] == 'Never'] but index not in df_never
df_neverH = df[df['smoking_status'] == 'Never']
# select index not in df_never
df_neverH = df_neverH[~df_neverH.index.isin(df_never.index)]

df_current = df[df['smoking_status']=='Current']
# Convert Current to 1 and Never to 0 in df_train
df_train['smoking_status'] = df_train['smoking_status'].replace({'Current':1,'Never':0})

train_df, test_df = train_test_split(df_train, test_size=0.3,random_state=random_state,stratify=df_train['smoking_status'])
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=random_state,stratify=train_df['smoking_status'])

x_train = train_df.drop(['smoking_status'],axis=1)
y_train = train_df['smoking_status']

x_val = val_df.drop(['smoking_status'],axis=1)
y_val = val_df['smoking_status']

x_test = test_df.drop(['smoking_status'],axis=1)
y_test = test_df['smoking_status']

splits = 5
with open('file_name', 'rb') as f:
    param_dict = pickle.load(f)

clf = clone(param_dict['init'])
X = pd.concat([x_train, y_val]) 
y = pd.concat([y_train, y_val])

_,_,_ = plot_roc_crossval_early_stop_df(X, y,clf,splits=splits,random_state=1996, title=f'5-fold cross validated ROC curve (All features)',name=f'../plot/init.pdf')

clf = clone(param_dict['init'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)
#calculate auc in test set
y_pred = clf.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, y_pred)
print(auc)

with open('file_name', 'rb') as f:
    model = pickle.load(f)


col_selected = list(model.support_)+[True]
df_train = df_train.iloc[:,col_selected]

pd.DataFrame({'Proteins':df_train.columns.to_list()[:-1]}).to_csv('../data/protein_list.csv',index=False)
# Convert Current to 1 and Never to 0 in df_train
df_train['smoking_status'] = df_train['smoking_status'].replace({'Current':1,'Never':0})

train_df, test_df = train_test_split(df_train, test_size=0.3,random_state=random_state,stratify=df_train['smoking_status'])
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=random_state,stratify=train_df['smoking_status'])

x_train = train_df.drop(['smoking_status'],axis=1)
y_train = train_df['smoking_status']

x_val = val_df.drop(['smoking_status'],axis=1)
y_val = val_df['smoking_status']

x_test = test_df.drop(['smoking_status'],axis=1)
y_test = test_df['smoking_status']
splits = 5
with open('file_name', 'rb') as f:
    param_dict = pickle.load(f)

clf = clone(param_dict['boruta'])
X = pd.concat([x_train, y_val])
y = pd.concat([y_train, y_val])

_,_,_ = plot_roc_crossval_early_stop_df(X, y,clf,splits=splits,random_state=1996, title=f'5-fold cross validated ROC curve (After Boruta selection)',name=f'../plot/boruta.pdf')
clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)
#calculate auc in test set
y_pred = clf.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, y_pred)
#SHAP
import shap
shap.initjs()

clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)

explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values[1],x_train,max_display=20,show=False)
plt.title('SHAP summary plot for Current smoker vs Non-smoker',size = 16)
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

y_pred = clf.predict(x_test, raw_score = True)

roc_auc_score(y_test, y_pred)

from sklearn.metrics import roc_curve, auc
fpr,tpr,thresholds = roc_curve(y_test, y_pred)

pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds})
#get tpr and thresholds when fpr is closest to 0.05
def find_nearest(fpr_cutoff):
    return tpr[abs(fpr-fpr_cutoff).argmin()], thresholds[abs(fpr-fpr_cutoff).argmin()]

print(find_nearest(0.01),find_nearest(0.05),find_nearest(0.1)
)
#calculate raw score for all
from joblib import Parallel, delayed
#import stratified kfold
from sklearn.model_selection import StratifiedKFold
early_stopping_rounds = 20
with open('file_name', 'rb') as f:
    param_dict = pickle.load(f)
all_preds_df = pd.DataFrame()
## For previous smokers
X_previous  = df[df['smoking_status'].isin(['Previous'])][x_train.columns]
X_neverH = df_neverH[x_train.columns]

previous_preds = []
neverH_preds = []
currentL_preds = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
for i, (train_idx, test_idx) in enumerate(cv.split(df_train, df_train['smoking_status'])):
    df_train_fold = df_train.iloc[train_idx]
    df_train_fold, df_val_fold = train_test_split(df_train_fold, test_size=0.2, random_state=random_state, shuffle=True,stratify=df_train_fold['smoking_status'])
    df_test_fold = df_train.iloc[test_idx]

    X_train_fold = df_train_fold.drop(['smoking_status'], axis=1)
    y_train_fold = df_train_fold['smoking_status']

    X_val_fold = df_val_fold.drop(['smoking_status'], axis=1)
    y_val_fold = df_val_fold['smoking_status']

    X_test_fold = df_test_fold.drop(['smoking_status'], axis=1)
    y_test_fold = df_test_fold['smoking_status']

    # Train model
    clf = clone(param_dict['boruta'])
    clf.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric=['auc'], early_stopping_rounds=early_stopping_rounds, verbose=False)

    # Predict
    y_pred_test = clf.predict(X_test_fold, raw_score = True)
    all_preds_df = pd.concat([all_preds_df,pd.DataFrame({'y_pred':y_pred_test}, index=X_test_fold.index)])

#Other 3 groups
clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)

# Predict previous
y_pred_previous = clf.predict(X_previous, raw_score = True)
previous_preds.append(y_pred_previous)

# Predict neverH
y_pred_neverH = clf.predict(X_neverH, raw_score = True)
neverH_preds.append(y_pred_neverH)



#make the index the same as x_previous
df_y_pred_previous = pd.DataFrame({'y_pred':np.mean(previous_preds,axis=0)}, index=X_previous.index)
df_y_pred_neverH = pd.DataFrame({'y_pred':np.mean(neverH_preds,axis=0)}, index=X_neverH.index)

#make tag column
df_y_pred_previous['tag'] = 'Previous'
df_y_pred_neverH['tag'] = 'NeverH'
#all_preds_df tag column equal to smoking_status
all_preds_df = all_preds_df.join(df['smoking_status'],how='inner')
all_preds_df['tag'] = all_preds_df['smoking_status']
#drop smoking_status column
all_preds_df = all_preds_df.drop(['smoking_status'],axis=1)

#concat df_y_pred_previous to all_preds_df
all_preds_df = pd.concat([all_preds_df,df_y_pred_previous,df_y_pred_neverH])
#add smoking status in
all_preds_df = all_preds_df.join(df['smoking_status'],how='inner')

#y_test = all_preds['smoking_status'] if Current 1 else 0
y_test = all_preds_df[all_preds_df['tag'].isin(['Current','Never'])]['tag'].replace({'Current':1,'Never':0}).to_list()
y_pred = all_preds_df[all_preds_df['tag'].isin(['Current','Never'])]['y_pred'].to_list()

#calculate thereshold for all
roc_auc_score(y_test, y_pred)

from sklearn.metrics import roc_curve, auc
fpr,tpr,thresholds = roc_curve(y_test, y_pred)

pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds})
#get tpr and thresholds when fpr is closest to 0.05
def find_nearest(fpr_cutoff):
    return tpr[abs(fpr-fpr_cutoff).argmin()], thresholds[abs(fpr-fpr_cutoff).argmin()]

#0.05 as cutoff
all_preds_df['pred_state'] = all_preds_df['y_pred'].apply(lambda x: 1 if x >= find_nearest(0.05)[1] else 0)

print(find_nearest(0.01),find_nearest(0.05),find_nearest(0.1)
)
#use sns to plot hist plot of both y_pred in blue and y_pred_previous in red
import seaborn as sns
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 4))

y_pred_never = all_preds_df[all_preds_df['smoking_status']=='Never']['y_pred'].to_list()
y_pred_current = all_preds_df[all_preds_df['smoking_status']=='Current']['y_pred'].to_list()
y_pred_previous = all_preds_df[all_preds_df['smoking_status']=='Previous']['y_pred'].to_list()



sns.histplot(y_pred_never, kde=True,bins=100, label='Never smokers',color='#00a087ff',edgecolor='none')

sns.histplot(y_pred_current, kde=True,bins=100, label='Current smokers',color='#b2182b',edgecolor='none')

sns.histplot(y_pred_previous, kde=True,bins=100, label='Previous smokers',color='#14213d',edgecolor='none')

#plot a vertical line as threshold
plt.axvline(x=find_nearest(0.05)[1], color='black', linestyle='--', label='threshold at fpr=0.05')
#set x label as 'score'
plt.xlabel('pSIN (Current smoker -->)',size=14)
plt.ylabel('Count',size=14)
plt.title('Distribution of pSIN in UKB',size=16,weight='bold')
#set x and y limit
plt.xlim(-9,11)
plt.ylim(0,1000)
plt.legend()

##Protein annotation
#-------------------
import pandas as pd
import requests
from tqdm import tqdm
#read the data from the csv file
proteins = pd.read_csv('file_name')['Proteins'].to_list()

# Replace 'your_app_name' with a name for your app when registering with STRING
caller_identity = 'your_app_name'
api_key = 'your_api_key'  # Optional: Include if you have an API key
species = '9606'  # Human, Homo sapiens

def get_annotations(protein):
    url = f"https://string-db.org/api/json/get_string_ids?identifiers={protein}&species={species}&caller_identity={caller_identity}&echo_query=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extract annotation if it exists in the response
        annotation_full = data[0].get('annotation') if data and 'annotation' in data[0] else 'No annotation found'
        # Keep only the first two sentences
        annotation = ' '.join(annotation_full.split('. ')[:2])
        return annotation
    else:
        return 'Failed to fetch data'

# Using a dictionary to store the results
results = {'Protein': [], 'Annotation': []}

for protein in tqdm(proteins):
    if protein == 'EBI3_IL27':
        protein = 'EBI3'
    annotation = get_annotations(protein)
    results['Protein'].append(protein)
    results['Annotation'].append(annotation)

# Convert the results into a DataFrame
df = pd.DataFrame(results)
#sort by protein names
df = df.sort_values('Protein')
df.to_csv('file_name',index=False)

#linear assocation models
#-------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

#Read metabolic age
all_preds_df = pd.read_csv('file_name',index_col=0)
#remove smoking_status column
all_preds_df = all_preds_df.drop(columns=['smoking_status'])
#read ukb data
ukb = pd.read_feather('file_name')
# ukb = pd.read_feather('/Users/xiao/Library/CloudStorage/OneDrive-Nexus365/DPhil/Projects/Smoking_score/01test/data/ukb_full_recoded_dataset_jun_22_2023.feather')
ukb = ukb.set_index('eid')
all_ncd = pd.read_feather("file_name")
#use eid as index
all_ncd.set_index('eid', inplace=True)
ukb = ukb.join(all_ncd[['prevalent_hypertension','prevalent_obesity','prevalent_diabetes']], how='inner')

# Convert birth_year and recruitment_date to datetime objects
ukb['birth_year'] = ukb['birth_year'].astype(int)
ukb['birth_month'] = pd.to_datetime(ukb['birth_month'], format='%B').dt.month
ukb['recruitment_date'] = pd.to_datetime(ukb['recruitment_date'])

# Calculate age in terms of months at recruitment date
age_in_yrs = ((ukb['recruitment_date'].dt.year - ukb['birth_year'])) + (ukb['recruitment_date'].dt.month - ukb['birth_month'])/12

# Create a new column for age at recruitment date
ukb['age_at_recruitment'] = age_in_yrs
ukb['age_squared'] = ukb['age_at_recruitment']**2

#AST/ALT ratio
ukb['AST_ALT_ratio'] = ukb['aspartate_aminotransferase']/ukb['alanine_aminotransferase']

#read in GlycA data
ngh = pd.read_feather('file_name')
#first column is the sample id
ngh = ngh.set_index('eid')[['GlycA']]
#merge metabolic age and ukb data
ukb = ukb.join(ngh, how='inner')

bio_type =[
'albumin',
'alkaline_phosphatase',
'apolipoprotein_A',
'apolipoprotein_B',
'calcium',
'glucose',
'hbA1c',
'LDL_direct',
'HDL_cholesterol',
'lipoprotein_A',
'phosphate',
'SHBG',
'testosterone',
'total_protein',
'urate',
'urea',
'vitamin_D',
'LTL_zadj',
'IGF1',
'C_reactive_protein',
'creatinine',
'cystatin_C',
'alanine_aminotransferase',
'aspartate_aminotransferase',
# 'AST_ALT_ratio',
'gamma_glutamyltransferase',
'total_bilirubin',
'cholesterol',
'triglycerides',
'GlycA'
]

bio_name = [
    'Albumin',
    'ALP',
    'APOA',
    'APOB',
    'Calcium',
    'Glucose',
    'HbA1c',
    'LDL cholesterol',
    'HDL cholesterol',
    'Lipoprotein A',
    'Phosphate',
    'SHBG',
    'Testosterone',
    'Total protein',
    'Urate',
    'Urea',
    'Vitamin D',
    'Telomere length',
    'IGF-1',
    'C-reactive protein',
    'Creatinine',
    'Cystatin C',
    'ALT',
    'AST',
    # 'AST/ALT',
    'GGT',
    'Total bilirubin',
    'Cholesterol',
    'Triglycerides',
    'GlycA'
]

blood_type = [
    'basophill_count',
    'eosinophill_count',
    'high_light_scatter_reticulocyte_count',
    'immature_reticulocyte_fraction',
    'lymphocyte_count',
    'mean_corpuscular_haemoglobin_concentration',
    'platelet_count',
    'monocyte_count',
    'neutrophill_count',
    'nucleated_red_blood_cell_count',
    'erythrocyte_count',
    'reticulocyte_count',
    'leukocyte_count',
    'haemoglobin_concentration'
]

blood_name = [
    'Basophil count',
    'Eosinophil count',
    'HL retic count',
    'IRF',
    'Lymphocyte count',
    'MCHC',
    'Platelet count',
    'Monocyte count',
    'Neutrophill count',
    'NRBC count',
    'Erythrocyte count',
    'Reticulocyte count',
    'Leukocyte count',
    'Hb conc'
]
phys_type = [
    'hand_grip_strength_right_normalized',
    'hand_grip_strength_left_normalized',
    # 'BMI',
    'FEV1_standardized',
    'FVC_standardized',
    # 'FEV1_FVC_ratio_z_score',
    # 'haemoglobin_concentration',
    'heel_bone_mineral_density',
    'pulse_wave_arterial_stiffness_index',
    'systolic_bp',
    'diastolic_bp',
    'identify_matches_mean_time',
    'fluid_intelligence',
    # '',
    'overall_health_poor',
    'usual_walking_pace_slow',
    'facial_aging_older',
    'tiredness_freq_everyday',
    'sleep_difficulty_usually',
    'sleep_hours_10',
    'prevalent_hypertension',
    'prevalent_obesity',
    'prevalent_diabetes',
    ]
phys_name = [
    'Hand grip strength (right)',
    'Hand grip strength (left)',
    # 'BMI',
    'Lung function (FEV1)',
    'Lung function (FVC)',
    # 'Lung function (FEV1/FVC)',
    # 'Haemoglobin concentration',
    'Heel bone mineral density',
    'Arterial stiffness index',
    'Systolic blood pressure',
    'Diastolic blood pressure',
    'Reaction time',
    'Fluid intelligence',
    # 'Frailty index (continuous)',
    'Poor self-rated health',
    'Slow walking pace',
    'Self-rated facial aging',
    'Tired/lethargic every day',
    'Frequent insomnia',
    'Sleep 10+ hours / day',
    'Hypertension',
    'Obesity',
    'Type II Diabetes'
]

binary_type = ['overall_health_poor',
    'usual_walking_pace_slow',
    'facial_aging_older',
    'tiredness_freq_everyday',
    'sleep_difficulty_usually',
    'sleep_hours_10',
    'prevalent_hypertension',
    'prevalent_obesity',
    'prevalent_diabetes',
    ]

#new column overall_health_poor if overall_health is 'Poor' else 0
ukb['overall_health_poor'] = np.where(ukb['overall_health'] == 'Poor', 1, 0)
#slow walking pace if walking_pace is 'Slow pace' else 0
ukb['usual_walking_pace_slow'] = np.where(ukb['usual_walking_pace'] == 'Slow pace', 1, 0)
#facial aging
ukb['facial_aging_older'] = np.where(ukb['facial_aging'] == 'Older than you are', 1, 0)
#new column tiredness_freq_everyday if tiredness_freq is 'Nearly every day' or 'More than half the days' else 0
ukb['tiredness_freq_everyday'] = np.where((ukb['tiredness_freq'] == 'Nearly every day') | (ukb['tiredness_freq'] == 'More than half the days'), 1, 0)
#sleep_difficulty
ukb['sleep_difficulty_usually'] = np.where(ukb['sleep_difficulty'] == 'Usually', 1, 0)
#sleep_hours
ukb['sleep_hours_10'] = np.where(ukb['sleep_hours'] >= 10, 1, 0)
ukb['coffee'] = ukb['coffee'].cat.reorder_categories(['0 cups/day', '0.5-1.9 cups/day', '2-2.9 cups/day','>=3 cups/day'], ordered=True)
ukb['tea'] = ukb['tea'].cat.reorder_categories(['<2 cups/day', '2-3.9 cups/day', '4-5.9 cups/day','>=6 cups/day'], ordered=True)
ukb['alcohol_freq'] = ukb['alcohol_freq'].cat.reorder_categories(['One to three times a month', 'Once or twice a week', 'Three or four times a week','Daily or almost daily'], ordered=True)
lm_models = {}
exposure = 'y_pred'
lm_models[exposure] = {'model1':{}}

co_var_list = ['recruitment_centre','ethnicity','alcohol_freq','education_years','IPAQ_activity_group','townsend_deprivation_index']
all_data = ukb.join(all_preds_df, how='inner')
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for tag in (bio_type + phys_type + blood_type):
    temp_df = all_data.copy()

    if tag == 'BMI':
        co_var_list_temp = [x for x in co_var_list if x != 'BMI']
    else:
        co_var_list_temp = co_var_list[:]

    if tag in ['systolic_bp','diastolic_bp']:
        temp_df = temp_df[~(temp_df['blood_pressure_meds'] == 'Yes')]
    else:
        temp_df = temp_df
    

    temp_df = temp_df[co_var_list_temp + [tag] + [exposure]]
    temp_df = temp_df.dropna()

    formula = f'{exposure} ~ {tag} + ' + ' + '.join(co_var_list_temp)

    if tag in binary_type:
        model = sm.formula.ols(formula=formula, data=temp_df).fit()
    else:
        #make tag standardized
        if tag != 'LTL_zadj':
            temp_df[tag] = (temp_df[tag] - temp_df[tag].mean()) / temp_df[tag].std()
        model = sm.formula.ols(formula=formula, data=temp_df).fit()
    
    lm_models[exposure]['model1'][tag] = model
  # Create a new figure and specify the layout using gridspec
import matplotlib.gridspec as gridspec
# import multipletests
from statsmodels.stats.multitest import multipletests

exposure = 'y_pred'
fig = plt.figure(figsize=(12, 9))
#add a title
fig.suptitle('Association between clinical biomarkers and risk factors to pSIN', fontsize=16, fontweight='bold', y=1.)
gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[2,2], height_ratios=[1])

# plot a: age and exposure
ax = plt.subplot(gs[0,0], aspect='auto')  # Span the entire left column
ax.set_title('a', fontweight='bold', loc='left')
# List to store hazard ratios and p-values
effect_size = []
p_values = []
ci_low_values = []
ci_high_values = []
event_counts = []

for tag in bio_type:
    # Get hazard ratio and p-value
    model = lm_models[exposure]['model1'][tag]

    es = model.params[tag]
    p = model.pvalues[tag]
    clow = model.conf_int().loc[tag, 0]
    chigh = model.conf_int().loc[tag, 1]
    event_count = model.nobs

    effect_size.append(es)
    p_values.append(p)
    ci_low_values.append(clow)
    ci_high_values.append(chigh)
    event_counts.append(event_count)

sorted_indices = np.argsort(effect_size)
sorted_hr_values = np.array(effect_size)[sorted_indices]
sorted_ci_low = np.array(ci_low_values)[sorted_indices]
sorted_ci_high = np.array(ci_high_values)[sorted_indices]
sorted_disease_list = np.array(bio_name)[sorted_indices]
sorted_pvals = np.array(p_values)[sorted_indices]
sorted_events = np.array(event_counts)[sorted_indices]


# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(sorted_pvals, method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')

bio_df = pd.DataFrame({'name':sorted_disease_list, 'pval':sorted_pvals, 'fdr_pval':fdr_corrected_pvals,'hr':sorted_hr_values, 'ci_low':sorted_ci_low, 'ci_high':sorted_ci_high})
# colors = np.where(sorted_pvals < 0.05, 'darkred', 'black')

# Create a horizontal line at y=0
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)

# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(sorted_hr_values)):
    plt.errorbar(
        sorted_hr_values[i], 
        i, 
        xerr=[[sorted_hr_values[i] - sorted_ci_low[i]], [sorted_ci_high[i] - sorted_hr_values[i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=colors[i]
    )

# Annotate the number of events to the right of the plot
index = 0.4
    
# Annotate the p-values to the right of the plot
plt.text(index, len(sorted_pvals) + 0.7, 'P-value', ha='left', va='center', fontweight='bold')
for i, count in enumerate(sorted_pvals):
    plt.text(index, i, f'{count:.2e}', ha='left', va='center', fontsize=9)

plt.xlabel('Beta')
plt.yticks(range(len(sorted_hr_values)), sorted_disease_list)
plt.title(f'Biochemical measurements', fontweight='bold')
# plt.xlim(-0.06,0.135)



# plot c: age and exposure
ax = plt.subplot(gs[0, 1], aspect='auto')  # Span the entire left column
ax.set_title('b', fontweight='bold', loc='left')
# List to store hazard ratios and p-values
effect_size = []
p_values = []
ci_low_values = []
ci_high_values = []
event_counts = []

for tag in phys_type:
    # Get hazard ratio and p-value
    model = lm_models[exposure]['model1'][tag]

    es = model.params[tag]
    p = model.pvalues[tag]
    clow = model.conf_int().loc[tag, 0]
    chigh = model.conf_int().loc[tag, 1]
    event_count = model.nobs

    effect_size.append(es)
    p_values.append(p)
    ci_low_values.append(clow)
    ci_high_values.append(chigh)
    event_counts.append(event_count)

sorted_indices = np.argsort(effect_size)
sorted_hr_values = np.array(effect_size)[sorted_indices]
sorted_ci_low = np.array(ci_low_values)[sorted_indices]
sorted_ci_high = np.array(ci_high_values)[sorted_indices]
sorted_disease_list = np.array(phys_name)[sorted_indices]
sorted_pvals = np.array(p_values)[sorted_indices]
sorted_events = np.array(event_counts)[sorted_indices]


# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(sorted_pvals, method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')

# colors = np.where(sorted_pvals < 0.05, 'darkred', 'black')
phy_df = pd.DataFrame({'name':sorted_disease_list, 'pval':sorted_pvals, 'fdr_pval':fdr_corrected_pvals,'hr':sorted_hr_values, 'ci_low':sorted_ci_low, 'ci_high':sorted_ci_high})

# Create a horizontal line at y=0
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)

# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(sorted_hr_values)):
    plt.errorbar(
        sorted_hr_values[i], 
        i, 
        xerr=[[sorted_hr_values[i] - sorted_ci_low[i]], [sorted_ci_high[i] - sorted_hr_values[i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=colors[i]
    )

# Annotate the number of events to the right of the plot
index = 1.4
    
# Annotate the p-values to the right of the plot
plt.text(index, len(sorted_pvals) - 0.1, 'P-value', ha='left', va='center', fontweight='bold')
for i, count in enumerate(sorted_pvals):
    plt.text(index, i, f'{count:.2e}', ha='left', va='center', fontsize=9)


plt.xlabel('Beta')
plt.yticks(range(len(sorted_hr_values)), sorted_disease_list)
plt.title(f'Clinical risk factors', fontweight='bold')
#set x axis range
# plt.xlim(-0.06,0.135)
plt.tight_layout()  

exposure = 'y_pred'
fig = plt.figure(figsize=(9, 6))
#add a title
fig.suptitle('Association between haematological measurements to pSIN', fontsize=16, fontweight='bold', y=1.)
gs = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1], height_ratios=[1])

# plot c: age and exposure
ax = plt.subplot(gs[0, 0], aspect='auto')  # Span the entire left column

# List to store hazard ratios and p-values
effect_size = []
p_values = []
ci_low_values = []
ci_high_values = []
event_counts = []

for tag in blood_type:
    # Get hazard ratio and p-value
    model = lm_models[exposure]['model1'][tag]

    es = model.params[tag]
    p = model.pvalues[tag]
    clow = model.conf_int().loc[tag, 0]
    chigh = model.conf_int().loc[tag, 1]
    event_count = model.nobs

    effect_size.append(es)
    p_values.append(p)
    ci_low_values.append(clow)
    ci_high_values.append(chigh)
    event_counts.append(event_count)

sorted_indices = np.argsort(effect_size)
sorted_hr_values = np.array(effect_size)[sorted_indices]
sorted_ci_low = np.array(ci_low_values)[sorted_indices]
sorted_ci_high = np.array(ci_high_values)[sorted_indices]
sorted_disease_list = np.array(blood_name)[sorted_indices]
sorted_pvals = np.array(p_values)[sorted_indices]
sorted_events = np.array(event_counts)[sorted_indices]


# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(sorted_pvals, method='fdr_bh')[1]
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')

# colors = np.where(sorted_pvals < 0.05, 'darkred', 'black')
blood_df = pd.DataFrame({'name':sorted_disease_list, 'pval':sorted_pvals, 'fdr_pval':fdr_corrected_pvals,'hr':sorted_hr_values, 'ci_low':sorted_ci_low, 'ci_high':sorted_ci_high})

# Create a horizontal line at y=0
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)

# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(sorted_hr_values)):
    plt.errorbar(
        sorted_hr_values[i], 
        i, 
        xerr=[[sorted_hr_values[i] - sorted_ci_low[i]], [sorted_ci_high[i] - sorted_hr_values[i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=colors[i]
    )

# Annotate the number of events to the right of the plot
index = 0.75

    
# Annotate the p-values to the right of the plot
plt.text(index, len(sorted_pvals) + 0., 'P-value', ha='left', va='center', fontweight='bold')
for i, count in enumerate(sorted_pvals):
    plt.text(index, i, f'{count:.2e}', ha='left', va='center', fontsize=9)


plt.xlabel('Beta')
plt.yticks(range(len(sorted_hr_values)), sorted_disease_list)

##calculating contribution
#-------------------------
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

param_dict = {}

n_trials = 150
early_stopping_rounds = 20
random_state = 1996

#read genotype data
genotype = pd.read_feather('file_name')
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
        
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric=[r2_score_lgbm], early_stopping_rounds=early_stopping_rounds, verbose=False)
        
            y_pred = model.predict(X_val_fold)
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
 
 import matplotlib.pyplot as plt
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

##Association with diseases
#-------------------------
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter
import math

#Read metabolic age
all_preds_df = pd.read_csv('file_name', index_col=0)
all_ncd = pd.read_feather("file_name")
#use eid as index
all_ncd.set_index('eid', inplace=True)
all_ncd = all_ncd.drop(columns=['recruitment_centre','recruitment_date'])

ukb = pd.read_feather('file_name')
ukb = ukb.set_index('eid')

ukb_exposure = pd.read_feather('file_name')
ukb_exposure = ukb_exposure.set_index('eid')

ukb['coffee'] = ukb['coffee'].cat.reorder_categories(['0 cups/day', '0.5-1.9 cups/day', '2-2.9 cups/day','>=3 cups/day'], ordered=True)
ukb['tea'] = ukb['tea'].cat.reorder_categories(['<2 cups/day', '2-3.9 cups/day', '4-5.9 cups/day','>=6 cups/day'], ordered=True)
ukb['alcohol_freq'] = ukb['alcohol_freq'].cat.reorder_categories(['One to three times a month', 'Once or twice a week', 'Three or four times a week','Daily or almost daily'], ordered=True)

#add smoking stop years
ukb_exposure['smoking_stop_years'] = ukb_exposure['recruitment_age'] - ukb_exposure['smoking_stop_age']
ukb = ukb.join(ukb_exposure['smoking_stop_years'], how='inner')

#if smoking status is Never, pack_years is 0
ukb['pack_years'] = np.where(ukb['smoking_status'] == 'Never', 0, ukb['pack_years'])

all_ncd = all_ncd.join(ukb, how='inner')

ncd_type = ['ACM', 'IHD', 'ischemic_stroke', 'all_stroke', 'COPD', 'liver', 'kidney', 'all_dementia', 'alzheimers', 'parkinsons', 'rheumatoid', 'macular', 'osteoporosis', 'osteoarthritis', 'PAD', 'asthma', 'CCF', 'IBD','vasc_dementia',"Cronh's",'Ulcerative Colitis']
ncd_name = ['All-cause mortality','Ischemic heart disease','Ischemic stroke','All stroke','COPD','Chronic liver disease','Chronic kidney disease','All-cause dementia',"Alzheimer's disease","Parkinson's disease",'Rheumatoid arthritis','Macular degeneration','Osteoporosis','Osteoarthritis','Peripheral artery\ndisease','Asthma','Congestive\ncardiac failure','Inflammatory\nbowel disease','Vascular dementia',"Cronh's disease",'Ulcerative colitis']

ncd_type = ncd_type + ['Lung', 'Colorectal', 'Pancreas', 'Kidney', 'Bladder', 'Head and neck', 'Oesophagus', 'Liver', 'Stomach']
ncd_name = ncd_name + ['Lung cancer', 'Colorectal cancer', 'Pancreatic cancer', 'Kidney cancer', 'Bladder cancer', 'Head and neck cancer', 'Oesophageal cancer', 'Liver cancer', 'Stomach cancer']

# ncd_type = ["IBD","Cronh's",'Ulcerative Colitis','Breast','Endometrium']
# ncd_name = ["'Inflammatory\nbowel disease'","Cronh's disease",'Ulcerative colitis','Breast cancer','Endometrial cancer']
cox_models = {}
protein_std = {}
co_var_list_all = ['recruitment_centre','ethnicity','alcohol_freq','education_years','BMI','IPAQ_activity_group','townsend_deprivation_index','smoking_status','pack_years']
sd = all_preds_df['y_pred'].std()

#drop smoking_status
if 'smoking_status' in all_preds_df.columns:
    all_preds_df_temp = all_preds_df.drop(columns=['smoking_status'])
else:
    all_preds_df_temp = all_preds_df

#model1
co_var_list = []
cox_models['model1'] = {}

all_data = all_ncd.join(all_preds_df_temp, how='inner')

cox_models = {}
protein_std = {}
co_var_list_all = ['recruitment_centre','ethnicity','alcohol_freq','education_years','BMI','IPAQ_activity_group','townsend_deprivation_index','smoking_status','pack_years']
sd = all_preds_df['y_pred'].std()

#drop smoking_status
if 'smoking_status' in all_preds_df.columns:
    all_preds_df_temp = all_preds_df.drop(columns=['smoking_status'])
else:
    all_preds_df_temp = all_preds_df

#model1
co_var_list = []
cox_models['model1'] = {}

all_data = all_ncd.join(all_preds_df_temp, how='inner')

all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)

#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

exposure = 'y_pred'
for ncd_tag,name in zip(ncd_type,ncd_name):

    if ncd_tag == 'ACM':
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
    else:
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = temp_df[~condition]
    
    # only females
    if ncd_tag in ['Endometrium','Breast']:
        temp_df = temp_df[temp_df['sex']=='Female']

    temp_df = temp_df[co_var_list_all+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()

    temp_df = temp_df[co_var_list+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    
    
    cph = CoxPHFitter()
    formula = f'{exposure}'
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula
    )
    cox_models['model1'][ncd_tag] = cph
    # # extract c index
    # c_ind = cph.concordance_index_

#model2
co_var_list = ['recruitment_centre','ethnicity','alcohol_freq','education_years','IPAQ_activity_group','townsend_deprivation_index','BMI']
cox_models['model2'] = {}

all_data = all_ncd.join(all_preds_df_temp, how='inner')

all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)

#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):

    if ncd_tag == 'ACM':
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
    else:
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = temp_df[~condition]

    # only females
    if ncd_tag in ['Endometrium','Breast']:
        temp_df = temp_df[temp_df['sex']=='Female']

    temp_df = temp_df[co_var_list_all+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()

    temp_df = temp_df[co_var_list+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    
    cph = CoxPHFitter()
    formula = f'{exposure} + '+ ' + '.join(co_var_list)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula
    )
    cox_models['model2'][ncd_tag] = cph
    # # extract c index
    # c_ind = cph.concordance_index_
#model3
cox_models['model3'] = {}

co_var_list = ['recruitment_centre','ethnicity','alcohol_freq','education_years','BMI','IPAQ_activity_group','townsend_deprivation_index','pack_years']
all_data = all_ncd.join(all_preds_df_temp, how='inner')

all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)

#remove recruitment centre 11022,11023 and 10003
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):
    print(name)
    if ncd_tag == 'ACM':
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
    else:
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = temp_df[~condition]

    # only females
    if ncd_tag in ['Endometrium','Breast']:
        temp_df = temp_df[temp_df['sex']=='Female']
        
    temp_df = temp_df[co_var_list_all+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()

    temp_df = temp_df[co_var_list+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]

    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}
    formula = f'{exposure} + '+ ' + '.join(co_var_list)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model3'][ncd_tag] = cph
# Create a new figure and specify the layout using gridspec
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import multipletests
from statsmodels.stats.multitest import multipletests

exposure = 'y_pred'

fig = plt.figure(figsize=(6, 20))
gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, width_ratios=[1], height_ratios=[24, 1, 1])

# plot a: age and sex
ax = plt.subplot(gs[0, 0], aspect='auto')  # Span the entire left column
# ax.set_title('a', fontweight='bold', loc='left')
# ax.set_title('Model 1', fontsize=9, loc='center')

odds_ratio = False

#Model1
# Extract hazard ratios and p-values from each model
df1 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio1':[],
                    'p_value1':[],
                    'ci_low1':[],
                    'ci_high1':[],
                    'event_counts1':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model1'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio1': hr,
            'p_value1': pval,
            'ci_low1': clow,
            'ci_high1': chigh,
            'event_counts1': events}
    df1 = pd.concat([df1, pd.DataFrame(new_row, index=[0])], ignore_index=True)

df1 = df1[df1['event_counts1'] >= 80]
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df1['p_value1'], method='fdr_bh')[1]
df1['p_value1'] = fdr_corrected_pvals

colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')
#add to the df1
df1['colors1'] = colors

#model2
# Extract hazard ratios and p-values from each model
df2 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio2':[],
                    'p_value2':[],
                    'ci_low2':[],
                    'ci_high2':[],
                    'event_counts2':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model2'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio2': hr,
            'p_value2': pval,
            'ci_low2': clow,
            'ci_high2': chigh,
            'event_counts2': events}
    df2 = pd.concat([df2, pd.DataFrame(new_row, index=[0])], ignore_index=True)

df2 = df2[df2['event_counts2'] >= 80]
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df2['p_value2'], method='fdr_bh')[1]
df2['p_value2'] = fdr_corrected_pvals
colors = np.where(fdr_corrected_pvals < 0.05, '#14213d', '#bdbdbd')
#add to the df1
df2['colors2'] = colors

#Model3
# Extract hazard ratios and p-values from each model
df3 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio3':[],
                    'p_value3':[],
                    'ci_low3':[],
                    'ci_high3':[],
                    'event_counts3':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model3'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df3
    new_row = {'ncd_name': name,
            'hazard_ratio3': hr,
            'p_value3': pval,
            'ci_low3': clow,
            'ci_high3': chigh,
            'event_counts3': events}
    df3 = pd.concat([df3, pd.DataFrame(new_row, index=[0])], ignore_index=True)

df3 = df3[df3['event_counts3'] >= 80]
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df3['p_value3'], method='fdr_bh')[1]
df3['p_value3'] = fdr_corrected_pvals
colors = np.where(fdr_corrected_pvals < 0.05, '#00a087ff', '#bdbdbd')
#add to the df1
df3['colors3'] = colors

#merge df1 and df2
df = df1.merge(df2, on='ncd_name')
df = df.merge(df3, on='ncd_name')
#remove in df where event_counts1 < 80
df = df[df['event_counts1'] >= 80]

# Sort the dataframes by hazard ratio
df = df.sort_values(by='hazard_ratio1', ascending=True)
#reset index
df = df.reset_index(drop=True)

# Create a horizontal line at y=0
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

interval = 0.15

# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(df['hazard_ratio1'])):
    plt.errorbar(
        x = df['hazard_ratio1'][i], 
        y = i+interval, 
        xerr=[[df['hazard_ratio1'][i] - df['ci_low1'][i]], [df['ci_high1'][i] - df['hazard_ratio1'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors1'][i]
    )
    plt.errorbar(
        x = df['hazard_ratio2'][i], 
        y = i-interval, 
        xerr=[[df['hazard_ratio2'][i] - df['ci_low2'][i]], [df['ci_high2'][i] - df['hazard_ratio2'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors2'][i]
    )


# Annotate the number of events to the right of the plot
index = 2.9

plt.text(index, len(df['event_counts1']) - 0.2, 'Events', ha='left', va='center', fontweight='bold')
for i, (count1, count2) in enumerate(zip(df['event_counts1'],df['event_counts2'])):
    plt.text(index, i+interval, f'{int(count1)}', ha='left', va='center', fontsize=9)
    plt.text(index, i-interval, f'{int(count2)}', ha='left', va='center', fontsize=9)

    
# Annotate the p-values to the right of the plot
plt.text(index+0.28, len(df['p_value1']) - 0.2, 'P-value', ha='left', va='center', fontweight='bold')
for i, (count1, count2) in enumerate(zip(df['p_value1'],df['p_value2'])):
    plt.text(index+0.28, i+interval, f'{count1:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.28, i-interval, f'{count2:.2e}', ha='left', va='center', fontsize=9)

#add legend
patch1 = mpatches.Patch(color='darkred', label='Model1',ls='--')
patch2 = mpatches.Patch(color='#14213d', label='Model2',ls='--')
patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')

plt.legend(handles=[patch1, patch2, patch4], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

plt.xlabel('Hazard ratio (95% CI)')
plt.yticks(range(len(df['hazard_ratio1'])), df['ncd_name'],fontsize=12)
plt.title(f'Multi-variate Cox model for pSIN\n (whole UKB population)', fontweight='bold',fontsize=14,y=1.02)

#set x axis limit
plt.xlim(0.5, 2.85)
plt.ylim(-0.7, len(df['hazard_ratio1'])-0.3)
cox_models = {}
co_var_list_all = ['recruitment_centre','ethnicity','alcohol_freq','education_years','IPAQ_activity_group','townsend_deprivation_index','BMI','pack_years','smoking_stop_years']


#drop smoking_status
if 'smoking_status' in all_preds_df.columns:
    all_preds_df_temp = all_preds_df.drop(columns=['smoking_status'])
else:
    all_preds_df_temp = all_preds_df

#model1
co_var_list = []
cox_models['model1'] = {}


all_data = all_ncd.join(all_preds_df_temp, how='inner')
all_data = all_data[all_data['smoking_status']=='Previous']
sd = all_data['y_pred'].std()

#remove recruitment centre 11022,11023 and 10003
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

exposure = 'y_pred'
for ncd_tag,name in zip(ncd_type,ncd_name):

    if ncd_tag == 'ACM':
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
    else:
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = temp_df[~condition]
    # only females
    if ncd_tag in ['Endometrium','Breast']:
        temp_df = temp_df[temp_df['sex']=='Female']
    temp_df = temp_df[co_var_list_all+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    temp_df = temp_df[co_var_list+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]

    cph = CoxPHFitter()
    formula = f'{exposure}'
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula
    )
    cox_models['model1'][ncd_tag] = cph
    
    # # extract c index
    # c_ind = cph.concordance_index_

#model2
co_var_list = ['recruitment_centre','ethnicity','alcohol_freq','education_years','IPAQ_activity_group','townsend_deprivation_index','BMI']
cox_models['model2'] = {}


all_data = all_ncd.join(all_preds_df_temp, how='inner')
all_data = all_data[all_data['smoking_status']=='Previous']
#remove recruitment centre 11022,11023 and 10003
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):

    if ncd_tag == 'ACM':
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
    else:
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = temp_df[~condition]
    # only females
    if ncd_tag in ['Endometrium','Breast']:
        temp_df = temp_df[temp_df['sex']=='Female']
    temp_df = temp_df[co_var_list_all+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    temp_df = temp_df[co_var_list+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    
    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}
    formula = f'{exposure} + '+ ' + '.join(co_var_list)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options

    )
    cox_models['model2'][ncd_tag] = cph
    
    # # extract c index
    # c_ind = cph.concordance_index_
#model3
cox_models['model3'] = {}

co_var_list = ['recruitment_centre','ethnicity','alcohol_freq','education_years','IPAQ_activity_group','townsend_deprivation_index','BMI','pack_years','smoking_stop_years']
all_data = all_ncd.join(all_preds_df_temp, how='inner')
all_data = all_data[all_data['smoking_status']=='Previous']

#remove recruitment centre 11022,11023 and 10003
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):
    print(name)
    if ncd_tag == 'ACM':
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
    else:
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = temp_df[~condition]
    # only females
    if ncd_tag in ['Endometrium','Breast']:
        temp_df = temp_df[temp_df['sex']=='Female']
    temp_df = temp_df[co_var_list_all+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    temp_df = temp_df[co_var_list+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]

    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}
    formula = f'{exposure} + '+ ' + '.join(co_var_list)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model3'][ncd_tag] = cph
    
# Create a new figure and specify the layout using gridspec
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import multipletests
from statsmodels.stats.multitest import multipletests

exposure = 'y_pred'

fig = plt.figure(figsize=(6, 20))
gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, width_ratios=[1], height_ratios=[24, 1, 1])

# plot a: age and sex
ax = plt.subplot(gs[0, 0], aspect='auto')  # Span the entire left column
# ax.set_title('a', fontweight='bold', loc='left')
# ax.set_title('Model 1', fontsize=9, loc='center')

odds_ratio = False

#Model1
# Extract hazard ratios and p-values from each model
df1 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio1':[],
                    'p_value1':[],
                    'ci_low1':[],
                    'ci_high1':[],
                    'event_counts1':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model1'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio1': hr,
            'p_value1': pval,
            'ci_low1': clow,
            'ci_high1': chigh,
            'event_counts1': events}
    df1 = pd.concat([df1, pd.DataFrame(new_row, index=[0])], ignore_index=True)

# df1 = df1[df1['event_counts1'] >= 80]
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df1['p_value1'], method='fdr_bh')[1]
df1['p_value1'] = fdr_corrected_pvals
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')
#add to the df1
df1['colors1'] = colors

#model2
# Extract hazard ratios and p-values from each model
df2 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio2':[],
                    'p_value2':[],
                    'ci_low2':[],
                    'ci_high2':[],
                    'event_counts2':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model2'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio2': hr,
            'p_value2': pval,
            'ci_low2': clow,
            'ci_high2': chigh,
            'event_counts2': events}
    df2 = pd.concat([df2, pd.DataFrame(new_row, index=[0])], ignore_index=True)

# df2 = df2[df2['event_counts2'] >= 80]
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df2['p_value2'], method='fdr_bh')[1]
df2['p_value2'] = fdr_corrected_pvals
colors = np.where(fdr_corrected_pvals < 0.05, '#14213d', '#bdbdbd')
#add to the df1
df2['colors2'] = colors

#Model3
# Extract hazard ratios and p-values from each model
df3 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio3':[],
                    'p_value3':[],
                    'ci_low3':[],
                    'ci_high3':[],
                    'event_counts3':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model3'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df3
    new_row = {'ncd_name': name,
            'hazard_ratio3': hr,
            'p_value3': pval,
            'ci_low3': clow,
            'ci_high3': chigh,
            'event_counts3': events}
    df3 = pd.concat([df3, pd.DataFrame(new_row, index=[0])], ignore_index=True)

# df3 = df3[df3['event_counts3'] >= 80]
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df3['p_value3'], method='fdr_bh')[1]
df3['p_value3'] = fdr_corrected_pvals
colors = np.where(fdr_corrected_pvals < 0.05, '#00a087ff', '#bdbdbd')
#add to the df1
df3['colors3'] = colors

#merge df1 and df2
df = df1.merge(df2, on='ncd_name')
df = df.merge(df3, on='ncd_name')
#remove in df where event_counts1 < 80
df = df[df['event_counts1'] >= 80]

# Sort the dataframes by hazard ratio
df = df.sort_values(by='hazard_ratio1', ascending=True)
#reset index
df = df.reset_index(drop=True)

# Create a horizontal line at y=0
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

interval = 0.3

# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(df['hazard_ratio1'])):
    plt.errorbar(
        x = df['hazard_ratio1'][i], 
        y = i+interval, 
        xerr=[[df['hazard_ratio1'][i] - df['ci_low1'][i]], [df['ci_high1'][i] - df['hazard_ratio1'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors1'][i]
    )
    plt.errorbar(
        x = df['hazard_ratio2'][i], 
        y = i, 
        xerr=[[df['hazard_ratio2'][i] - df['ci_low2'][i]], [df['ci_high2'][i] - df['hazard_ratio2'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors2'][i]
    )

    plt.errorbar(
        x = df['hazard_ratio3'][i], 
        y = i-interval, 
        xerr=[[df['hazard_ratio3'][i] - df['ci_low3'][i]], [df['ci_high3'][i] - df['hazard_ratio3'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors3'][i]
    )

# Annotate the number of events to the right of the plot
index = 2.9

plt.text(index, len(df['event_counts1']) - 0.3, 'Events', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['event_counts1'],df['event_counts2'],df['event_counts3'])):

    plt.text(index, i+interval, f'{int(count1)}', ha='left', va='center', fontsize=9)
    plt.text(index, i, f'{int(count2)}', ha='left', va='center', fontsize=9)
    plt.text(index, i-interval, f'{int(count3)}', ha='left', va='center', fontsize=9)

    
# Annotate the p-values to the right of the plot
plt.text(index+0.28, len(df['p_value1']) - 0.3, 'P-value', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['p_value1'],df['p_value2'],df['p_value3'])):
    plt.text(index+0.28, i+interval, f'{count1:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.28, i, f'{count2:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.28, i-interval, f'{count3:.2e}', ha='left', va='center', fontsize=9)

#add legend
patch1 = mpatches.Patch(color='darkred', label='Model1',ls='--')
patch2 = mpatches.Patch(color='#14213d', label='Model2',ls='--')
patch3 = mpatches.Patch(color='#00a087ff', label='Model3',ls='--')
patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')

plt.legend(handles=[patch1, patch2, patch3, patch4], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

plt.xlabel('Hazard ratio (95% CI)')
plt.yticks(range(len(df['hazard_ratio1'])), df['ncd_name'],fontsize=12)
plt.title(f'Association between pSIN and major diseases and mortality\n (in previous smokers)', fontweight='bold', fontsize=14,y=1.04)

#set x axis limit
plt.xlim(0.5, 2.85)
plt.ylim(-0.7, len(df['hazard_ratio1'])-0.3)
cox_models = {}
co_var_list_all = ['recruitment_centre','ethnicity','alcohol_freq','education_years','IPAQ_activity_group','townsend_deprivation_index','BMI','pack_years']

#drop smoking_status
if 'smoking_status' in all_preds_df.columns:
    all_preds_df_temp = all_preds_df.drop(columns=['smoking_status'])
else:
    all_preds_df_temp = all_preds_df

#model1
co_var_list = []
cox_models['model1'] = {}

all_data = all_ncd.join(all_preds_df_temp, how='inner')
all_data = all_data[all_data['smoking_status']=='Current']
sd = all_data['y_pred'].std()

#remove recruitment centre 11022,11023 and 10003
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

exposure = 'y_pred'
for ncd_tag,name in zip(ncd_type,ncd_name):

    if ncd_tag == 'ACM':
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
    else:
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = temp_df[~condition]

    #remove nan
    temp_df = temp_df[co_var_list_all+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    temp_df = temp_df[co_var_list+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]


    scipy_minimize_options = {'step_size': 0.1}
    cph = CoxPHFitter()
    formula = f'{exposure}'
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model1'][ncd_tag] = cph

    # # extract c index
    # c_ind = cph.concordance_index_

#model2
co_var_list = ['recruitment_centre','ethnicity','alcohol_freq','education_years','IPAQ_activity_group','townsend_deprivation_index','BMI']
cox_models['model2'] = {}

all_data = all_ncd.join(all_preds_df_temp, how='inner')
all_data = all_data[all_data['smoking_status']=='Current']
#remove recruitment centre 11022,11023 and 10003
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]

for ncd_tag,name in zip(ncd_type,ncd_name):

    if ncd_tag == 'ACM':
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
    else:
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = temp_df[~condition]

    #remove nan
    temp_df = temp_df[co_var_list_all+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    temp_df = temp_df[co_var_list+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    
    cph = CoxPHFitter()
    formula = f'{exposure} + '+ ' + '.join(co_var_list)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model2'][ncd_tag] = cph

    # # extract c index
    # c_ind = cph.concordance_index_
#model3
cox_models['model3'] = {}
co_var_list = ['recruitment_centre','ethnicity','alcohol_freq','education_years','IPAQ_activity_group','townsend_deprivation_index','BMI','pack_years']
all_data = all_ncd.join(all_preds_df_temp, how='inner')
all_data = all_data[all_data['smoking_status']=='Current']


#remove recruitment centre 11022,11023 and 10003
all_data['recruitment_centre'] = all_data['recruitment_centre'].astype(str)
all_data = all_data[~all_data['recruitment_centre'].isin(['11022','11023','10003'])]


for ncd_tag,name in zip(ncd_type,ncd_name):
    print(name)
    if ncd_tag == 'ACM':
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
    else:
        temp_df = all_data
        #fill na with 0 in {ncd_tag}_event
        temp_df[f'{ncd_tag}_event'] = temp_df[f'{ncd_tag}_event'].fillna(0)
        condition = (all_data[f'incident_{ncd_tag}'] == 0) & (all_data[f'{ncd_tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df = temp_df[~condition]

    #remove nan
    temp_df = temp_df[co_var_list_all+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]
    temp_df = temp_df.dropna()
    temp_df = temp_df[co_var_list+[f'{ncd_tag}_event',f'{ncd_tag}_survival_time',exposure]]

    cph = CoxPHFitter()
    scipy_minimize_options = {'step_size': 0.1}
    formula = f'{exposure} + '+ ' + '.join(co_var_list)
    cph.fit(
        temp_df, 
        duration_col=f'{ncd_tag}_survival_time', 
        event_col=f'{ncd_tag}_event',
        formula=formula,
        fit_options=scipy_minimize_options
    )
    cox_models['model3'][ncd_tag] = cph

# Create a new figure and specify the layout using gridspec
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import multipletests
from statsmodels.stats.multitest import multipletests

exposure = 'y_pred'

fig = plt.figure(figsize=(6, 20))
gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, width_ratios=[1], height_ratios=[24, 1, 1])

# plot a: age and sex
ax = plt.subplot(gs[0, 0], aspect='auto')  # Span the entire left column
# ax.set_title('a', fontweight='bold', loc='left')
# ax.set_title('Model 1', fontsize=9, loc='center')

odds_ratio = False

#Model1
# Extract hazard ratios and p-values from each model
df1 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio1':[],
                    'p_value1':[],
                    'ci_low1':[],
                    'ci_high1':[],
                    'event_counts1':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model1'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio1': hr,
            'p_value1': pval,
            'ci_low1': clow,
            'ci_high1': chigh,
            'event_counts1': events}
    df1 = pd.concat([df1, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df1['p_value1'], method='fdr_bh')[1]
df1['p_value1'] = fdr_corrected_pvals
colors = np.where(fdr_corrected_pvals < 0.05, 'darkred', '#bdbdbd')
#add to the df1
df1['colors1'] = colors

#model2
# Extract hazard ratios and p-values from each model
df2 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio2':[],
                    'p_value2':[],
                    'ci_low2':[],
                    'ci_high2':[],
                    'event_counts2':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model2'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df1
    new_row = {'ncd_name': name,
            'hazard_ratio2': hr,
            'p_value2': pval,
            'ci_low2': clow,
            'ci_high2': chigh,
            'event_counts2': events}
    df2 = pd.concat([df2, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df2['p_value2'], method='fdr_bh')[1]
df2['p_value2'] = fdr_corrected_pvals
colors = np.where(fdr_corrected_pvals < 0.05, '#14213d', '#bdbdbd')
#add to the df1
df2['colors2'] = colors

#Model3
# Extract hazard ratios and p-values from each model
df3 = pd.DataFrame({'ncd_name':[],
                    'hazard_ratio3':[],
                    'p_value3':[],
                    'ci_low3':[],
                    'ci_high3':[],
                    'event_counts3':[]})

for ncd_tag,name in zip(ncd_type,ncd_name):
    model = cox_models['model3'][ncd_tag]
    hr = math.exp(model.summary['coef'][exposure]*sd)
    pval = model.summary['p'][exposure]
    clow = math.exp(model.summary['coef lower 95%'][exposure]*sd)
    chigh = math.exp(model.summary['coef upper 95%'][exposure]*sd)
    events = model.event_observed.sum()

    if odds_ratio:
        hr = np.log2(np.exp(hr))
        clow = np.log2(np.exp(clow))
        chigh = np.log2(np.exp(chigh))
    
    #append to the df3
    new_row = {'ncd_name': name,
            'hazard_ratio3': hr,
            'p_value3': pval,
            'ci_low3': clow,
            'ci_high3': chigh,
            'event_counts3': events}
    df3 = pd.concat([df3, pd.DataFrame(new_row, index=[0])], ignore_index=True)
# Define colors based on fdr-corrected p-values
fdr_corrected_pvals = multipletests(df3['p_value3'], method='fdr_bh')[1]
df3['p_value3'] = fdr_corrected_pvals
colors = np.where(fdr_corrected_pvals < 0.05, '#00a087ff', '#bdbdbd')
#add to the df1
df3['colors3'] = colors

#merge df1 and df2
df = df1.merge(df2, on='ncd_name')
df = df.merge(df3, on='ncd_name')

#remove in df where event_counts1 < 80
df = df[df['event_counts1'] >= 80]
# Sort the dataframes by hazard ratio
df = df.sort_values(by='hazard_ratio1', ascending=True)
#reset index
df = df.reset_index(drop=True)

# Create a horizontal line at y=0
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)

interval = 0.3

# Plot the hazard ratios and confidence intervals with colored dots
for i in range(len(df['hazard_ratio1'])):
    plt.errorbar(
        x = df['hazard_ratio1'][i], 
        y = i+interval, 
        xerr=[[df['hazard_ratio1'][i] - df['ci_low1'][i]], [df['ci_high1'][i] - df['hazard_ratio1'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors1'][i]
    )
    plt.errorbar(
        x = df['hazard_ratio2'][i], 
        y = i, 
        xerr=[[df['hazard_ratio2'][i] - df['ci_low2'][i]], [df['ci_high2'][i] - df['hazard_ratio2'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors2'][i]
    )

    plt.errorbar(
        x = df['hazard_ratio3'][i], 
        y = i-interval, 
        xerr=[[df['hazard_ratio3'][i] - df['ci_low3'][i]], [df['ci_high3'][i] - df['hazard_ratio3'][i]]],
        fmt='s', 
        markersize=4, 
        capsize=2, 
        color=df['colors3'][i]
    )

# Annotate the number of events to the right of the plot
index = 2.9

plt.text(index, len(df['event_counts1']) - 0.3, 'Events', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['event_counts1'],df['event_counts2'],df['event_counts3'])):

    plt.text(index, i+interval, f'{int(count1)}', ha='left', va='center', fontsize=9)
    plt.text(index, i, f'{int(count2)}', ha='left', va='center', fontsize=9)
    plt.text(index, i-interval, f'{int(count3)}', ha='left', va='center', fontsize=9)

    
# Annotate the p-values to the right of the plot
plt.text(index+0.3, len(df['p_value1']) - 0.3, 'P-value', ha='left', va='center', fontweight='bold')
for i, (count1, count2,count3) in enumerate(zip(df['p_value1'],df['p_value2'],df['p_value3'])):
    plt.text(index+0.3, i+interval, f'{count1:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.3, i, f'{count2:.2e}', ha='left', va='center', fontsize=9)
    plt.text(index+0.3, i-interval, f'{count3:.2e}', ha='left', va='center', fontsize=9)

#add legend
patch1 = mpatches.Patch(color='darkred', label='Model1',ls='--')
patch2 = mpatches.Patch(color='#14213d', label='Model2',ls='--')
patch3 = mpatches.Patch(color='#00a087ff', label='Model3',ls='--')
patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')

plt.legend(handles=[patch1, patch2, patch3, patch4], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

plt.xlabel('Hazard ratio (95% CI)')
plt.yticks(range(len(df['hazard_ratio1'])), df['ncd_name'],fontsize=12)
plt.title(f'Association between pSIN and major diseases and mortality\n (in current smokers)', fontweight='bold', fontsize=14,y=1.04)

plt.xlim(0.5, 2.85)
plt.ylim(-0.7, len(df['hazard_ratio1'])-0.3)

## Cumulative incidence plot
#--------------------------------
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
from lifelines.plotting import add_at_risk_counts
import matplotlib.patches as mpatches
### get top, middle, bottom 10%
def cut_by_thresholds(series):
    top_10_threshold = series.quantile(0.9)
    bottom_10_threshold = series.quantile(0.1)
    median_10_top = series.quantile(0.55)
    median_10_bottom = series.quantile(0.45)
 
    categories = pd.Series(['Other'] * len(series), index=series.index)
    categories[series > top_10_threshold] = 'Top 10%'
    categories[(series <= median_10_top) & (series >= median_10_bottom)] = 'Median 10%'
    categories[series < bottom_10_threshold] = 'Bottom 10%'
 
    return categories

### get top, middle, bottom 25%
def cut_by_thresholds25(series):
    top_10_threshold = series.quantile(0.75)
    bottom_10_threshold = series.quantile(0.25)
    median_10_top = series.quantile(0.65)
    median_10_bottom = series.quantile(0.35)
 
    categories = pd.Series(['Other'] * len(series), index=series.index)
    categories[series > top_10_threshold] = 'Top 25%'
    categories[(series <= median_10_top) & (series >= median_10_bottom)] = 'Median 25%'
    categories[series < bottom_10_threshold] = 'Bottom 25%'
 
    return categories

all_ncd = pd.read_feather("file_name")
#use eid as index
all_ncd.set_index('eid', inplace=True)

all_preds_df = pd.read_csv('file_name', index_col=0)

all_preds_df = all_preds_df.join(all_ncd, how='left')

ukb = pd.read_feather('file_name')
ukb = ukb.set_index('eid')
ukb = ukb[['sex', 'birth_year','birth_month','recruitment_date','smoking_status','pack_years']]

# Convert birth_year and recruitment_date to datetime objects
ukb['birth_year'] = ukb['birth_year'].astype(int)
ukb['birth_month'] = pd.to_datetime(ukb['birth_month'], format='%B').dt.month
ukb['recruitment_date'] = pd.to_datetime(ukb['recruitment_date'])

# Calculate age in terms of months at recruitment date
age_in_yrs = ((ukb['recruitment_date'].dt.year - ukb['birth_year'])) + (ukb['recruitment_date'].dt.month - ukb['birth_month'])/12

# Create a new column for age at recruitment date
ukb['age_at_recruitment'] = age_in_yrs

#if smoking status is Never, pack_years is 0
ukb['pack_years'] = np.where(ukb['smoking_status'] == 'Never', 0, ukb['pack_years'])

all_preds_df = all_preds_df.join(ukb[['age_at_recruitment','pack_years']], how='inner')


ncd_type = ['Lung', 'COPD', 'Head and neck', 'PAD', 'Oesophagus', 'Bladder', 'ACM', 'CCF', 'liver', 'all_stroke', 'ischemic_stroke', "Cronh's",'IHD', 'Kidney', 'IBD', 'kidney', 'rheumatoid', 'Colorectal', 'asthma', 'Pancreas', 'all_dementia', 'osteoporosis', 'osteoarthritis', 'macular', 'alzheimers', 'parkinsons','vasc_dementia','Ulcerative Colitis']
ncd_name = ['Lung cancer', 'COPD', 'Head and neck cancer', 'Peripheral artery\ndisease', 'Oesophageal cancer', 'Bladder cancer', 'All-cause mortality', 'Congestive\ncardiac failure', 'Chronic liver disease', 'All stroke', 'Ischemic stroke',"Cronh's disease", 'Ischemic heart disease', 'Kidney cancer', 'Inflammatory\nbowel disease', 'Chronic kidney disease', 'Rheumatoid arthritis', 'Colorectal cancer', 'Asthma', 'Pancreatic cancer', 'All-cause dementia', 'Osteoporosis', 'Osteoarthritis', 'Macular degeneration', "Alzheimer's disease", "Parkinson's disease",'Vascular dementia','Ulcerative colitis']
kmf_dict = {}

ignore_type = ['Kidney','Pancreatic','all_dementia','osteoporosis','osteoarthritis','macular','alzheimers','vasc_dementia','Ulcerative Colitis']
# Create a figure and subplots
fig, axes = plt.subplots(4, 5, figsize=(16, 12))
fig.suptitle(f"Cumulative incidence of major diseases and mortality by quartile of pSIN\n (In UKB overall population)",y=1, fontsize=20, fontweight='bold')
fig.text(-0.01, 0.5, 'Cumulative incidence',ha='center', rotation='vertical',size=14)
fig.text(0.5,-0.01, 'Age',ha='center',size=14)

# Flatten the axes array to simplify indexing
axes = axes.flatten()
i=0

for tag,name in zip(ncd_type,ncd_name):    
    if tag in ignore_type:
        continue
    kmf_dict[tag] = {}

    ax = axes[i]
    #remove prevalent
    if tag == 'ACM':
        temp_df_all = all_preds_df.copy()
        #fill na with 0 in {tag}_event
        temp_df_all[f'{tag}_event'] = temp_df_all[f'{tag}_event'].fillna(0)
    else:
        temp_df_all = all_preds_df.copy()
        #fill na with 0 in {tag}_event
        temp_df_all[f'{tag}_event'] = temp_df_all[f'{tag}_event'].fillna(0)
        condition = (all_preds_df[f'incident_{tag}'] != 1) & (all_preds_df[f'{tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df_all = temp_df_all[~condition]


    # temp_df_all = all_preds_df[~all_preds_df[f'incident_{tag}'].isin(['Prevalent diagnosis'])]

    temp_df_all['y_pred_decile'] = cut_by_thresholds25(temp_df_all['y_pred'])

    temp_df = temp_df_all[temp_df_all['y_pred_decile'].isin(['Top 25%','Median 25%','Bottom 25%'])]

    color_list = ['#ef3b2c','#2171b5','#66c2a4']
    color = iter(color_list)

    T = temp_df[f'{tag}_survival_time']
    #convert to delta datetime
    # T = pd.to_timedelta(T)
    #convert to year
    # T = T/365.25
    T = T/365.25+temp_df['age_at_recruitment']
    # T = temp_df['age_at_recruitment']

    E = temp_df[f'{tag}_event']
    # #fill the missing value with 0


    groups = temp_df[f'y_pred_decile']

    c=next(color)
    ix = (groups == 'Top 25%')
    kmf = KaplanMeierFitter()
    kmf.fit(T[ix], E[ix], label='Top 25%')
    kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75))
    kmf_dict[tag]['Top 25%'] = kmf

    c=next(color)
    ix = (groups == 'Median 25%')
    kmf = KaplanMeierFitter()
    kmf.fit(T[ix], E[ix], label='Median 25%')
    kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75)  )
    kmf_dict[tag]['Median 25%'] = kmf

    c=next(color)
    ix = (groups == 'Bottom 25%')
    kmf = KaplanMeierFitter()
    kmf.fit(T[ix], E[ix], label='Bottom 25%')
    kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75)  )
    kmf_dict[tag]['Bottom 25%'] = kmf

    #set the x axis limit
    # ax.set_xlim(40,80)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    n = int(temp_df_all[f'{tag}_event'].sum())
    ax.set_title(f'{name} (n={n})',fontsize=12)
    ax.legend().set_visible(False)

    #remove the legend
    i+=1
#remove the last two empty plot
# fig.delaxes(axes[-1])
# fig.delaxes(axes[-2])
# fig.delaxes(axes[-3])
# Add a legend to the bottom
handles, labels = ax.get_legend_handles_labels()

#add legend
color = iter(color_list)
c = next(color)
patch1 = mpatches.Patch(color=c, label='Top 25% pSIN',ls='-')
c = next(color)
patch2 = mpatches.Patch(color=c, label='Median 25% pSIN',ls='-')
c = next(color)
patch3 = mpatches.Patch(color=c, label='Bottom 25% pSIN',ls='-')
# patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')
fig.legend(handles=[patch1, patch2, patch3], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12)

plt.tight_layout()
#save the plot with ppi=1200
# cumulative_density
cum_den_df = pd.DataFrame()
for tag,name in zip(ncd_type, ncd_name):
    if tag not in kmf_dict:
        continue
    if not kmf_dict[tag]:
        continue
    for quartile in ['Top 25%','Median 25%','Bottom 25%']:
        line = pd.DataFrame(kmf_dict[tag][quartile].cumulative_density_at_times([50,55,60,65,70,75])).T
        #reset row index
        line = line.reset_index(drop=False)
        #insert tag as the first column
        line.insert(0,'Outcome',name)
        #rename all columns
        line.columns = ['Outcome','PredSmokingScore quartiles','Age 50','Age 55','Age 60','Age 65','Age 70','Age 75']
        #concatenate line to cum_den_df
        cum_den_df = pd.concat([cum_den_df, line])

#reset index
cum_den_df = cum_den_df.reset_index(drop=True)
cum_den_df = cum_den_df.replace('\n',' ', regex=True)

cum_den_df.to_csv('file_name', index=False)

target_values = [50, 55, 60, 65, 70, 75]

at_risk_df = pd.DataFrame()
for tag,name in zip(ncd_type, ncd_name):
    if tag not in kmf_dict:
        continue
    if not kmf_dict[tag]:
        continue
    
    for quartile in kmf_dict[tag].keys():
        event_table = pd.DataFrame(kmf_dict[tag][quartile].event_table)
        #reset row index
        event_table = event_table.reset_index(drop=False)

        # Calculate the absolute differences for each target value
        for target in target_values:
            event_table[f'diff_{target}'] = abs(event_table['event_at'] - target)

        # Find the index with the minimum absolute difference for each target
        indexes = [event_table[f'diff_{target}'].idxmin() for target in target_values]

        # Select rows based on the found indexes
        result_df = event_table.loc[indexes]

        line = result_df[['at_risk']].T
        line = line.reset_index(drop=True)
        line.insert(0,'PredSmokingScore quartiles',quartile)
        line.insert(0,'Outcome',name)
        line.columns = ['Outcome','PredSmokingScore quartiles','Age 50','Age 55','Age 60','Age 65','Age 70','Age 75']
        at_risk_df = pd.concat([at_risk_df, line])
#reset index
at_risk_df = at_risk_df.reset_index(drop=True)
at_risk_df = at_risk_df.replace('\n',' ', regex=True)

at_risk_df.to_csv('file_name', index=False)

kmf_dict_current = {}
et_dict_current = {}

keep_type = ['Lung','PAD','COPD','ACM','osteoporosis','IHD']
keep_name = ['Lung cancer','Peripheral artery disease','COPD','All-cause mortality','Osteoporosis','Ischemic heart disease']


# Create a figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(8, 5))
fig.suptitle(f"Cumulative incidence of chronic diseases and mortality by \nquartile of pSIN (In current smokers)",y=1, fontsize=14,weight='bold')
fig.text(-0.01, 0.3, 'Cumulative incidence',ha='center', rotation='vertical',size=14)
fig.text(0.5,-0.01, 'Age',ha='center',size=14)

# Flatten the axes array to simplify indexing
axes = axes.flatten()
i=0

for tag,name in zip(keep_type,keep_name): 

    kmf_dict_current[tag] = {}
    et_dict_current[tag] = {}

    #remove prevalent
    if tag == 'ACM':
        temp_df_all = all_preds_df
        #fill na with 0 in {tag}_event
        temp_df_all[f'{tag}_event'] = temp_df_all[f'{tag}_event'].fillna(0)
    else:
        temp_df_all = all_preds_df
        #fill na with 0 in {tag}_event
        temp_df_all[f'{tag}_event'] = temp_df_all[f'{tag}_event'].fillna(0)
        condition = (all_preds_df[f'incident_{tag}'] != 1) & (all_preds_df[f'{tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df_all = temp_df_all[~condition]

    temp_df = temp_df_all[temp_df_all['smoking_status']=='Current']
    n = int(temp_df[f'{tag}_event'].sum())
    if n <= 80:
        continue
    ax = axes[i]

    temp_df['y_pred_decile'] = cut_by_thresholds25(temp_df['y_pred'])

    temp_df = temp_df[temp_df['y_pred_decile'].isin(['Top 25%','Median 25%','Bottom 25%'])]

    color_list = ['#ef3b2c','#2171b5','#66c2a4']
    color = iter(color_list)

    T = temp_df[f'{tag}_survival_time']
    #convert to delta datetime
    # T = pd.to_timedelta(T)
    #convert to year
    # T = T/365.25
    T = T/365.25+temp_df['age_at_recruitment']
    # T = temp_df['age_at_recruitment']

    E = temp_df[f'{tag}_event']



    groups = temp_df[f'y_pred_decile']

    c=next(color)
    ix = (groups == 'Top 25%')
    kmf = KaplanMeierFitter()
    kmf.fit(T[ix], E[ix], label='Top 25%')
    kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75))
    kmf_dict_current[tag]['Top 25%'] = kmf

    c=next(color)
    ix = (groups == 'Median 25%')
    kmf = KaplanMeierFitter()
    kmf.fit(T[ix], E[ix], label='Median 25%')
    kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75)  )
    kmf_dict_current[tag]['Median 25%'] = kmf

    c=next(color)
    ix = (groups == 'Bottom 25%')
    kmf = KaplanMeierFitter()
    kmf.fit(T[ix], E[ix], label='Bottom 25%')
    kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75)  )
    kmf_dict_current[tag]['Bottom 25%'] = kmf

    #set the x axis limit
    # ax.set_xlim(40,80)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    n = int(temp_df[f'{tag}_event'].sum())
    ax.set_title(f'{name} (n={n})',fontsize=12)
    ax.legend().set_visible(False)

    #remove the legend
    i+=1
#remove the last two empty plot
# fig.delaxes(axes[-1])
# fig.delaxes(axes[-2])
# fig.delaxes(axes[-3])
# Add a legend to the bottom
handles, labels = ax.get_legend_handles_labels()

#add legend
color = iter(color_list)
c = next(color)
patch1 = mpatches.Patch(color=c, label='Top 25% pSIN',ls='-')
c = next(color)
patch2 = mpatches.Patch(color=c, label='Median 25% pSIN',ls='-')
c = next(color)
patch3 = mpatches.Patch(color=c, label='Bottom 25% pSIN',ls='-')
# patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')
fig.legend(handles=[patch1, patch2, patch3], loc='lower center', bbox_to_anchor=(0.5, -0.105), ncol=4, fontsize=12)

plt.tight_layout()
#save the plot with ppi=1200

# cumulative_density
cum_den_df_current = pd.DataFrame()
for tag,name in zip(ncd_type, ncd_name):
    if tag not in kmf_dict_current:
        continue
    if not kmf_dict_current[tag]:
        continue
    for quartile in ['Top 25%','Median 25%','Bottom 25%']:
        line = pd.DataFrame(kmf_dict_current[tag][quartile].cumulative_density_at_times([50,55,60,65,70,75])).T
        #reset row index
        line = line.reset_index(drop=False)
        #insert tag as the first column
        line.insert(0,'Outcome',name)
        #rename all columns
        line.columns = ['Outcome','PredSmokingScore quartiles','Age 50','Age 55','Age 60','Age 65','Age 70','Age 75']
        #concatenate line to cum_den_df
        cum_den_df_current = pd.concat([cum_den_df_current, line])

#reset index
cum_den_df_current = cum_den_df_current.reset_index(drop=True)
cum_den_df_current = cum_den_df_current.replace('\n',' ', regex=True)
cum_den_df_current.to_csv('file_name', index=False)

target_values = [50, 55, 60, 65, 70, 75]

at_risk_df_current = pd.DataFrame()
for tag,name in zip(ncd_type, ncd_name):
    if tag not in kmf_dict_current:
        continue
    if not kmf_dict_current[tag]:
        continue
    
    for quartile in kmf_dict_current[tag].keys():
        event_table = pd.DataFrame(kmf_dict_current[tag][quartile].event_table)
        #reset row index
        event_table = event_table.reset_index(drop=False)

        # Calculate the absolute differences for each target value
        for target in target_values:
            event_table[f'diff_{target}'] = abs(event_table['event_at'] - target)

        # Find the index with the minimum absolute difference for each target
        indexes = [event_table[f'diff_{target}'].idxmin() for target in target_values]

        # Select rows based on the found indexes
        result_df = event_table.loc[indexes]

        line = result_df[['at_risk']].T
        line = line.reset_index(drop=True)
        line.insert(0,'PredSmokingScore quartiles',quartile)
        line.insert(0,'Outcome',name)
        line.columns = ['Outcome','PredSmokingScore quartiles','Age 50','Age 55','Age 60','Age 65','Age 70','Age 75']
        at_risk_df_current = pd.concat([at_risk_df_current, line])
#reset index
at_risk_df_current = at_risk_df_current.reset_index(drop=True)
at_risk_df_current = at_risk_df_current.replace('\n',' ', regex=True)

at_risk_df_current.to_csv('file_name', index=False)

kmf_dict_prev = {}
et_dict_prev = {}

ncd_type = ['Lung', 'COPD', 'Head and neck', 'PAD', 'Oesophagus', 'Bladder', 'ACM', 'CCF', 'liver', 'all_stroke', 'ischemic_stroke', "Cronh's",'IHD', 'Kidney', 'IBD', 'kidney', 'rheumatoid', 'Colorectal', 'asthma', 'Pancreas', 'all_dementia', 'osteoporosis', 'osteoarthritis', 'macular', 'alzheimers', 'parkinsons','vasc_dementia','Ulcerative Colitis']

ignore_type = ['Colorectal','macular','rheumatoid','all_dementia','osteoporosis','alzheimers', 'parkinsons','IBD','Ulcerative Colitis','vasc_dementia', "Cronh's"]
# Create a figure and subplots
fig, axes = plt.subplots(4, 3, figsize=(8, 10))
fig.suptitle(f"Cumulative incidence of major diseases and mortality by \nquartile of pSIN (In previous smokers)",y=1, fontsize=14,weight='bold')
fig.text(-0.01, 0.5, 'Cumulative incidence',ha='center', rotation='vertical',size=14)
fig.text(0.5,-0.01, 'Age',ha='center',size=14)

# Flatten the axes array to simplify indexing
axes = axes.flatten()
i=0

for tag,name in zip(ncd_type,ncd_name): 
    if tag in ignore_type:
        continue
    kmf_dict_prev[tag] = {}
    et_dict_prev[tag] = {}

    #remove prevalent
    if tag == 'ACM':
        temp_df_all = all_preds_df
        #fill na with 0 in {tag}_event
        temp_df_all[f'{tag}_event'] = temp_df_all[f'{tag}_event'].fillna(0)
    else:
        temp_df_all = all_preds_df
        #fill na with 0 in {tag}_event
        temp_df_all[f'{tag}_event'] = temp_df_all[f'{tag}_event'].fillna(0)
        condition = (all_preds_df[f'incident_{tag}'] != 1) & (all_preds_df[f'{tag}_event'] == 1)
        # Use boolean indexing to remove rows where the condition is True
        temp_df_all = temp_df_all[~condition]
    temp_df = temp_df_all[temp_df_all['smoking_status']=='Previous']
    n = int(temp_df[f'{tag}_event'].sum())
    if n <= 80:
        continue
    ax = axes[i]
    temp_df['y_pred_decile'] = cut_by_thresholds25(temp_df['y_pred'])

    temp_df = temp_df[temp_df['y_pred_decile'].isin(['Top 25%','Median 25%','Bottom 25%'])]

    color_list = ['#ef3b2c','#2171b5','#66c2a4']
    color = iter(color_list)

    T = temp_df[f'{tag}_survival_time']

    T = T/365.25+temp_df['age_at_recruitment']
    # T = temp_df['age_at_recruitment']

    E = temp_df[f'{tag}_event']
  

    groups = temp_df[f'y_pred_decile']

    c=next(color)
    ix = (groups == 'Top 25%')
    kmf = KaplanMeierFitter()
    kmf.fit(T[ix], E[ix], label='Top 25%')
    kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75))
    kmf_dict_prev[tag]['Top 25%'] = kmf

    c=next(color)
    ix = (groups == 'Median 25%')
    kmf = KaplanMeierFitter()
    kmf.fit(T[ix], E[ix], label='Median 25%')
    kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75)  )
    kmf_dict_prev[tag]['Median 25%'] = kmf

    c=next(color)
    ix = (groups == 'Bottom 25%')
    kmf = KaplanMeierFitter()
    kmf.fit(T[ix], E[ix], label='Bottom 25%')
    kmf.plot_cumulative_density(color=c,ax=ax,loc=slice(45,75)  )
    kmf_dict_prev[tag]['Bottom 25%'] = kmf

    #set the x axis limit
    # ax.set_xlim(40,80)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    n = int(temp_df[f'{tag}_event'].sum())
    ax.set_title(f'{name} (n={n})',fontsize=12)
    ax.legend().set_visible(False)

    #remove the legend
    i+=1
#remove the last two empty plot
# fig.delaxes(axes[-1])
# fig.delaxes(axes[-2])
# fig.delaxes(axes[-3])
# Add a legend to the bottom
handles, labels = ax.get_legend_handles_labels()

#add legend
color = iter(color_list)
c = next(color)
patch1 = mpatches.Patch(color=c, label='Top 25% pSIN',ls='-')
c = next(color)
patch2 = mpatches.Patch(color=c, label='Median 25% pSIN',ls='-')
c = next(color)
patch3 = mpatches.Patch(color=c, label='Bottom 25% pSIN',ls='-')
# patch4 = mpatches.Patch(color='#bdbdbd', label='Not significant',ls='--')
fig.legend(handles=[patch1, patch2, patch3], loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=12)

plt.tight_layout()
#save the plot with ppi=1200
# plt.savefig('../plot/cumulative_risks_previous_population.png',dpi=1200,bbox_inches='tight')

# cumulative_density
cum_den_df_prev = pd.DataFrame()
for tag,name in zip(ncd_type, ncd_name):
    if tag not in kmf_dict_prev.keys():
        continue
    if not kmf_dict_prev[tag].keys():
        continue
    for quartile in ['Top 25%','Median 25%','Bottom 25%']:
        line = pd.DataFrame(kmf_dict_prev[tag][quartile].cumulative_density_at_times([50,55,60,65,70,75])).T
        #reset row index
        line = line.reset_index(drop=False)
        #insert tag as the first column
        line.insert(0,'Outcome',name)
        #rename all columns
        line.columns = ['Outcome','PredSmokingScore quartiles','Age 50','Age 55','Age 60','Age 65','Age 70','Age 75']
        #concatenate line to cum_den_df_prev
        cum_den_df_prev = pd.concat([cum_den_df_prev, line])

#reset index
cum_den_df_prev = cum_den_df_prev.reset_index(drop=True)
cum_den_df_prev = cum_den_df_prev.replace('\n',' ', regex=True)
cum_den_df_prev.to_csv('file_name', index=False)

target_values = [50, 55, 60, 65, 70, 75]

at_risk_df_prev = pd.DataFrame()
for tag,name in zip(ncd_type, ncd_name):
    if tag not in kmf_dict_prev:
        continue
    if not kmf_dict_prev[tag]:
        continue
    
    for quartile in kmf_dict_prev[tag].keys():
        event_table = pd.DataFrame(kmf_dict_prev[tag][quartile].event_table)
        #reset row index
        event_table = event_table.reset_index(drop=False)

        # Calculate the absolute differences for each target value
        for target in target_values:
            event_table[f'diff_{target}'] = abs(event_table['event_at'] - target)

        # Find the index with the minimum absolute difference for each target
        indexes = [event_table[f'diff_{target}'].idxmin() for target in target_values]

        # Select rows based on the found indexes
        result_df = event_table.loc[indexes]

        line = result_df[['at_risk']].T
        line = line.reset_index(drop=True)
        line.insert(0,'PredSmokingScore quartiles',quartile)
        line.insert(0,'Outcome',name)
        line.columns = ['Outcome','PredSmokingScore quartiles','Age 50','Age 55','Age 60','Age 65','Age 70','Age 75']
        at_risk_df_prev = pd.concat([at_risk_df_prev, line])
#reset index
at_risk_df_prev = at_risk_df_prev.reset_index(drop=True)
at_risk_df_prev = at_risk_df_prev.replace('\n',' ', regex=True)

at_risk_df_prev.to_csv('file_name', index=False)