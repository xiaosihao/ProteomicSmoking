#Calculate pSIN and SHAP
#------------------------
import pandas as pd  # import libraries
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score,precision_score
from sklearn.model_selection import train_test_split  # split data
from lightgbm import LGBMRegressor
import numpy as np
from machine_learning import *
import pickle
random_state = 1996

#read csv file
olink_data = pd.read_feather('file_name')  # read data
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

_,_,_ = plot_roc_crossval_early_stop_df(X, y,clf,splits=splits,random_state=1996, title=f'5-fold cross validated ROC curve (All features)',name=f'../plot/init.pdf')  # plot results

clf = clone(param_dict['init'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)  # fit model
#calculate auc in test set
y_pred = clf.predict_proba(x_test)[:,1]  # predict
auc = roc_auc_score(y_test, y_pred)
print(auc)

with open('file_name', 'rb') as f:
    model = pickle.load(f)


col_selected = list(model.support_)+[True]
df_train = df_train.iloc[:,col_selected]

pd.DataFrame({'Proteins':df_train.columns.to_list()[:-1]}).to_csv('../data/protein_list.csv',index=False)  # save to csv
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