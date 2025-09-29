# %%
import pandas as pd  # import libraries
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score,precision_score, average_precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split  # split data
from lightgbm import LGBMRegressor
import numpy as np
from machine_learning import *
import pickle
random_state = 1996


# %%
#read csv file
olink_data = pd.read_feather("PathName/FileName")  # read data
olink_data = olink_data.set_index('eid')

# ukb = pd.read_feather("PathName/FileName")
ukb = pd.read_feather("PathName/FileName")
ukb = ukb.set_index('eid')

#remove overlapping columns
ukb = ukb.drop(columns = [i for i in ukb.columns if i in olink_data.columns])

df = ukb.join(olink_data,how='inner')

#remove samples with missing smoking status
df = df[~df['smoking_status'].isna()]


# %%
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


# %%
pro = olink_data.columns.to_list()
#keep samples that are current in smoking_status and those who are No in ever_smoked
df_train = pd.concat([df_current,df_never])[pro+['smoking_status']]
# df_train = df[df['smoking_status']!='Previous'][pro+['smoking_status']]
#reset smoking status datatype to category to only have 2 categories
df_train.smoking_status = df_train.smoking_status.astype('str')
df_train.smoking_status = df_train.smoking_status.astype('category')


# %%
def metrics(y_test,y_pred):
    auc = roc_auc_score(y_test, y_pred)
    f1 = precision_score(y_test, y_pred.round())
    ap = average_precision_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred.round())
    return auc,f1,ap,ba

# %%
def metrics_5cv(X,y,clf,splits=5,random_state=1996):
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    auc_scores = []
    f1_scores = []
    ap_scores = []
    ba_scores = []

    for i,(train_idx, val_idx) in enumerate(cv.split(X,y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        model = clone(clf)

        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)  # fit model
        y_pred = model.predict_proba(X_val_fold)[:,1]  # predict
        
        auc,f1,ap,ba = metrics(y_val_fold,y_pred)
        auc_scores.append(auc)
        f1_scores.append(f1)
        ap_scores.append(ap)
        ba_scores.append(ba)

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    mean_ap = np.mean(ap_scores)
    std_ap = np.std(ap_scores)

    mean_ba = np.mean(ba_scores)
    std_ba = np.std(ba_scores)

    print(f'AUC: {mean_auc} +/- {std_auc}')
    print(f'F1: {mean_f1} +/- {std_f1}')
    print(f'AP: {mean_ap} +/- {std_ap}')
    print(f'BA: {mean_ba} +/- {std_ba}')

# %%
with open('../data/boruta_clf.p', 'rb') as f:
    model = pickle.load(f)


col_selected = list(model.support_)+[True]
df_train = df_train.iloc[:,col_selected]

pd.DataFrame({'Proteins':df_train.columns.to_list()[:-1]}).to_csv("PathName/FileName",index=False)  # save to csv


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
with open('../data/param_dict_clf.p', 'rb') as f:
    param_dict = pickle.load(f)


# %% [markdown]
# # Undersampling

# %%
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)

#undersample the test set to match 1, 0 ratio of current to never smokers
rus = RandomUnderSampler(random_state=random_state)
X_resampled, y_resampled = rus.fit_resample(x_test, y_test)

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_resampled, y_resampled,random_state=random_state)
    
    
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
# # Sex specific

# %%
#Add sex column from ukb to x_test using left join
x_test_sex = x_test.join(ukb['sex'],how='left')

#get male data
x_test_male = x_test_sex[x_test_sex['sex'] == 'Male']
#drop sex column
x_test_male = x_test_male.drop('sex', axis=1)
# select row in y_test where index is in x_test_male index
y_test_male = y_test[y_test.index.isin(x_test_male.index)]

#get female data
x_test_female = x_test_sex[x_test_sex['sex'] == 'Female']
#drop sex column
x_test_female = x_test_female.drop('sex', axis=1)
# select row in y_test where index is in x_test_male index
y_test_female = y_test[y_test.index.isin(x_test_female.index)]



# %%
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

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
    X_bootstrap, y_bootstrap = resample(x_test_male, y_test_male,random_state=random_state)
    
    
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

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(x_test_female, y_test_female,random_state=random_state)
    
    
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
X_resampled, y_resampled = rus.fit_resample(x_test_male, y_test_male)

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_resampled, y_resampled,random_state=random_state)
    
    
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
X_resampled, y_resampled = rus.fit_resample(x_test_female, y_test_female)

# Number of bootstrap iterations
num_iterations = 100
auc_scores = []
f1_scores = []
ap_scores = []
ba_scores = []

# Perform bootstrapping for 100 iterations
for _ in range(num_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_resampled, y_resampled,random_state=random_state)
    
    
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


