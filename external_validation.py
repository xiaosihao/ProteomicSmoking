# %%
import pandas as pd  # import libraries
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score,precision_score, average_precision_score, balanced_accuracy_score,roc_curve,auc
from sklearn.model_selection import train_test_split  # split data
from lightgbm import LGBMRegressor
import numpy as np
from machine_learning import *
import pickle
random_state = 1996

def metrics(y_test,y_pred):
    auc = roc_auc_score(y_test, y_pred)
    f1 = precision_score(y_test, y_pred.round())
    ap = average_precision_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred.round())
    return auc,f1,ap,ba


# %%
#read ckb data
ckb_data = pd.read_feather("PathName/FileName")  # read data
ckb_data = ckb_data.set_index('csid')

#read ckb baseline data
ckb = pd.read_feather("PathName/FileName")
ckb = ckb.set_index('csid')
#only those in ckb_data index
ckb = ckb.loc[ckb_data.index,:]

# %%
#new column called smoking_status where 1,2 in smoking_category is Never, 3 is Previous, 4 is Current
ckb['smoking_status'] = ckb['smoking_category'].replace([1,2],'Never').replace([3],'Previous').replace([4],'Current')
ckb['recruitment_age'] = ckb['age_at_study_date_x100']/100

# %%
df_cbcgrps= ckb[ckb['smoking_status'].isin(['Never','Current'])]
df_cbcgrps['smoking_status'] = df_cbcgrps['smoking_status'].astype(str)
df_cbcgrps= df_cbcgrps[df_cbcgrps['is_female']==1]
#convert smoking status to str
df_cbcgrps['smoking_status'] = df_cbcgrps['smoking_status'].astype(str)
#only keep needed columns
df_cbcgrps = df_cbcgrps[['recruitment_age','bmi_calc','met','alcohol_category','highest_education','smoking_status']]
df_cbcgrps.to_csv("PathName/FileName",index=False)  # save to csv

# %%
# df_cbcgrps= ckb[ckb['smoking_status'].isin(['Never','Current'])]
df_cbcgrps = ckb.copy()
# df_cbcgrps= df_cbcgrps[df_cbcgrps['is_female']==1]
#convert smoking status to str
df_cbcgrps['smoking_status'] = df_cbcgrps['smoking_status'].astype(str)
#only keep needed columns
df_cbcgrps = df_cbcgrps[['recruitment_age','bmi_calc','met','alcohol_category','highest_education','smoking_status','is_female']]
df_cbcgrps.to_csv("PathName/FileName",index=False)

# %%
ckb_temp = ckb[ckb['is_female']==1]

df_temp = ckb_temp[ckb_temp['smoking_status']=='Current']
# df_temp = ckb_temp.copy()
# Calculate value counts of 'alcohol_freq'
value_counts = df_temp['highest_education'].value_counts()

# Calculate percentages
percentages = (value_counts / len(df_temp)) * 100
percentages = percentages.round(2)  # Round to two decimal places

# Create DataFrame
result_df = pd.DataFrame({'Value': value_counts.index, 'Value_Counts':value_counts,'Value_Counts_Percentage': percentages})

#combine value counts and percentages column
result_df['combined'] = result_df['Value_Counts'].astype(str) + ' (' + result_df['Value_Counts_Percentage'].astype(str) + '%)'
#sort by index
result_df = result_df.sort_index()

print(result_df[['combined']])

# %%
urban = ckb_temp[ckb_temp['smoking_status']=='Previous']
# urban = ckb_temp.copy()
perc = urban['region_is_urban'].value_counts()[0]/urban['region_is_urban'].value_counts().sum()*100
f"{urban['region_is_urban'].value_counts()[0]}({perc:.3f}%)"

# %%
#read csv file
olink_data = pd.read_feather("PathName/FileName")
olink_data = olink_data.set_index('eid')

# ukb = pd.read_feather("PathName/FileName")
ukb = pd.read_feather("PathName/FileName")
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

pro = olink_data.columns.to_list()
#keep samples that are current in smoking_status and those who are No in ever_smoked
df_train = pd.concat([df_current,df_never])[pro+['smoking_status']]
# df_train = df[df['smoking_status']!='Previous'][pro+['smoking_status']]
#reset smoking status datatype to category to only have 2 categories
df_train.smoking_status = df_train.smoking_status.astype('str')
df_train.smoking_status = df_train.smoking_status.astype('category')

# Convert Current to 1 and Never to 0 in df_train
df_train['smoking_status'] = df_train['smoking_status'].replace({'Current':1,'Never':0})


# %%
with open('../../02_4classification/data/boruta_clf.p', 'rb') as f:
    model = pickle.load(f)


col_selected = list(model.support_)+[True]
df_train = df_train.iloc[:,col_selected]


# %%
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

# %%
with open('../../02_4classification/data/param_dict_clf.p', 'rb') as f:
    param_dict = pickle.load(f)


# %%
clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)  # fit model
#calculate auc in test set
y_pred = clf.predict(x_test, raw_score = True)  # predict
auc = roc_auc_score(y_test, y_pred)
auc


# %%
from sklearn.metrics import precision_score, recall_score, f1_score
cutoff = -1.2910091125564642

# Convert predicted probabilities to binary predictions
y_pred_binary = (y_pred > cutoff)  # Choose an appropriate threshold (e.g., 0.5 for binary classification)

# Calculate precision
precision = precision_score(y_test, y_pred_binary)

# Calculate recall
recall = recall_score(y_test, y_pred_binary)

# Calculate F1-score
f1 = f1_score(y_test, y_pred_binary)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# %% [markdown]
# # Both sex

# %%
#merge ckb_data with smoking_status from ckb
ckb_val = ckb_data.join(ckb['smoking_status'],how='inner')

#only man
ckb_val = ckb_val.join(ckb['is_female'],how='inner')
# ckb_val = ckb_val[ckb_val['is_female']==0]
#drop is_female from ckb_val
ckb_val = ckb_val.drop(['is_female'],axis=1)

# select columns in df_train
ckb_val = ckb_val[df_train.columns]
#only never and current smokers
ckb_val = ckb_val[ckb_val['smoking_status'].isin(['Never','Current'])]
# Convert Current to 1 and Never to 0 in ckb_val
ckb_val['smoking_status'] = ckb_val['smoking_status'].replace({'Current':1,'Never':0})

X_ckb = ckb_val.drop(['smoking_status'],axis=1)
y_ckb = ckb_val['smoking_status']

# %%
from sklearn.utils import resample
clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_ckb, y_ckb, random_state=random_state)
    
    
    # Predict probabilities on original X_ckb
    y_pred = clf.predict_proba(X_bootstrap)[:,1]
    
    auc,f1,ap,ba = metrics(y_bootstrap,y_pred)
    auc_scores.append(auc)
    f1_scores.append(f1)
    ap_scores.append(ap)
    ba_scores.append(ba)


# Calculate mean and standard deviation of AUC values
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

mean_ap = np.mean(ap_scores)
std_ap = np.std(ap_scores)

mean_ba = np.mean(ba_scores)
std_ba = np.std(ba_scores)

#print results
print(f'AUC: {auc:.3f} +/- {std_auc:.3e}')
print(f'F1: {mean_f1:.3f} +/- {std_f1:.3e}')
print(f'AP: {mean_ap:.3f} +/- {std_ap:.3e}')
print(f'BA: {mean_ba:.3f} +/- {std_ba:.3e}')

# %%
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)
#undersample the test set to match 1, 0 ratio of current to never smokers
rus = RandomUnderSampler(random_state=random_state)
X_resampled, y_resampled = rus.fit_resample(X_ckb, y_ckb)

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_resampled, y_resampled, random_state=random_state)
    
    
    # Predict probabilities on original X_ckb
    y_pred = clf.predict_proba(X_bootstrap)[:,1]
    
    auc,f1,ap,ba = metrics(y_bootstrap,y_pred)
    auc_scores.append(auc)
    f1_scores.append(f1)
    ap_scores.append(ap)
    ba_scores.append(ba)


# Calculate mean and standard deviation of AUC values
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

mean_ap = np.mean(ap_scores)
std_ap = np.std(ap_scores)

mean_ba = np.mean(ba_scores)
std_ba = np.std(ba_scores)

#print results
print(f'AUC: {auc:.3f} +/- {std_auc:.3e}')
print(f'F1: {mean_f1:.3f} +/- {std_f1:.3e}')
print(f'AP: {mean_ap:.3f} +/- {std_ap:.3e}')
print(f'BA: {mean_ba:.3f} +/- {std_ba:.3e}')

# %% [markdown]
# # Male

# %%
#merge ckb_data with smoking_status from ckb
ckb_val = ckb_data.join(ckb['smoking_status'],how='inner')

#only man
ckb_val = ckb_val.join(ckb['is_female'],how='inner')
ckb_val = ckb_val[ckb_val['is_female']==0]
#drop is_female from ckb_val
ckb_val = ckb_val.drop(['is_female'],axis=1)

# select columns in df_train
ckb_val = ckb_val[df_train.columns]
#only never and current smokers
ckb_val = ckb_val[ckb_val['smoking_status'].isin(['Never','Current'])]
# Convert Current to 1 and Never to 0 in ckb_val
ckb_val['smoking_status'] = ckb_val['smoking_status'].replace({'Current':1,'Never':0})

X_ckb = ckb_val.drop(['smoking_status'],axis=1)
y_ckb = ckb_val['smoking_status']

# %%
from sklearn.utils import resample
clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_ckb, y_ckb, random_state=random_state)
    
    
    # Predict probabilities on original X_ckb
    y_pred = clf.predict_proba(X_bootstrap)[:,1]
    
    auc,f1,ap,ba = metrics(y_bootstrap,y_pred)
    auc_scores.append(auc)
    f1_scores.append(f1)
    ap_scores.append(ap)
    ba_scores.append(ba)


# Calculate mean and standard deviation of AUC values
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

mean_ap = np.mean(ap_scores)
std_ap = np.std(ap_scores)

mean_ba = np.mean(ba_scores)
std_ba = np.std(ba_scores)

#print results
print(f'AUC: {auc:.3f} +/- {std_auc:.3f}')
print(f'F1: {mean_f1:.3f} +/- {std_f1:.3f}')
print(f'AP: {mean_ap:.3f} +/- {std_ap:.3f}')
print(f'BA: {mean_ba:.3f} +/- {std_ba:.3f}')

# %%
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)
#undersample the test set to match 1, 0 ratio of current to never smokers
rus = RandomUnderSampler(random_state=random_state)
X_resampled, y_resampled = rus.fit_resample(X_ckb, y_ckb)

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_resampled, y_resampled, random_state=random_state)
    
    
    # Predict probabilities on original X_ckb
    y_pred = clf.predict_proba(X_bootstrap)[:,1]
    
    auc,f1,ap,ba = metrics(y_bootstrap,y_pred)
    auc_scores.append(auc)
    f1_scores.append(f1)
    ap_scores.append(ap)
    ba_scores.append(ba)


# Calculate mean and standard deviation of AUC values
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

mean_ap = np.mean(ap_scores)
std_ap = np.std(ap_scores)

mean_ba = np.mean(ba_scores)
std_ba = np.std(ba_scores)

#print results
print(f'AUC: {auc:.3f} +/- {std_auc:.3e}')
print(f'F1: {mean_f1:.3f} +/- {std_f1:.3e}')
print(f'AP: {mean_ap:.3f} +/- {std_ap:.3e}')
print(f'BA: {mean_ba:.3f} +/- {std_ba:.3e}')



# %% [markdown]
# # Female

# %%
#merge ckb_data with smoking_status from ckb
ckb_val = ckb_data.join(ckb['smoking_status'],how='inner')

#only man
ckb_val = ckb_val.join(ckb['is_female'],how='inner')
ckb_val = ckb_val[ckb_val['is_female']==1]
#drop is_female from ckb_val
ckb_val = ckb_val.drop(['is_female'],axis=1)

# select columns in df_train
ckb_val = ckb_val[df_train.columns]
#only never and current smokers
ckb_val = ckb_val[ckb_val['smoking_status'].isin(['Never','Current'])]
# Convert Current to 1 and Never to 0 in ckb_val
ckb_val['smoking_status'] = ckb_val['smoking_status'].replace({'Current':1,'Never':0})

X_ckb = ckb_val.drop(['smoking_status'],axis=1)
y_ckb = ckb_val['smoking_status']

# %%
from sklearn.utils import resample
clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_ckb, y_ckb, random_state=random_state)
    
    
    # Predict probabilities on original X_ckb
    y_pred = clf.predict_proba(X_bootstrap)[:,1]
    
    auc,f1,ap,ba = metrics(y_bootstrap,y_pred)
    auc_scores.append(auc)
    f1_scores.append(f1)
    ap_scores.append(ap)
    ba_scores.append(ba)


# Calculate mean and standard deviation of AUC values
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

mean_ap = np.mean(ap_scores)
std_ap = np.std(ap_scores)

mean_ba = np.mean(ba_scores)
std_ba = np.std(ba_scores)

#print results
print(f'AUC: {auc:.3f} +/- {std_auc:.3f}')
print(f'F1: {mean_f1:.3f} +/- {std_f1:.3f}')
print(f'AP: {mean_ap:.3f} +/- {std_ap:.3f}')
print(f'BA: {mean_ba:.3f} +/- {std_ba:.3f}')

# %%
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)
#undersample the test set to match 1, 0 ratio of current to never smokers
rus = RandomUnderSampler(random_state=random_state)
X_resampled, y_resampled = rus.fit_resample(X_ckb, y_ckb)

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_resampled, y_resampled, random_state=random_state)
    
    
    # Predict probabilities on original X_ckb
    y_pred = clf.predict_proba(X_bootstrap)[:,1]
    
    auc,f1,ap,ba = metrics(y_bootstrap,y_pred)
    auc_scores.append(auc)
    f1_scores.append(f1)
    ap_scores.append(ap)
    ba_scores.append(ba)


# Calculate mean and standard deviation of AUC values
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

mean_ap = np.mean(ap_scores)
std_ap = np.std(ap_scores)

mean_ba = np.mean(ba_scores)
std_ba = np.std(ba_scores)

#print results
print(f'AUC: {auc:.3f} +/- {std_auc:.3e}')
print(f'F1: {mean_f1:.3f} +/- {std_f1:.3e}')
print(f'AP: {mean_ap:.3f} +/- {std_ap:.3e}')
print(f'BA: {mean_ba:.3f} +/- {std_ba:.3e}')

