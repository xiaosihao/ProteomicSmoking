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

with open('../../02_4classification/data/param_dict_clf.p', 'rb') as f:
    param_dict = pickle.load(f)

col_selected = list(model.support_)+[True]
df_train = df_train.iloc[:,col_selected]


# %%
#In validation dataset
from itertools import cycle


fpr_dic = {}
tpr_dic = {}
roc_auc_dic = {}


name = 'pSIN'

#select df_train where index is in eid_sex
df_train_temp = df_train.copy()

train_df, test_df = train_test_split(df_train_temp, test_size=0.3,random_state=random_state,stratify=df_train_temp['smoking_status'])
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=random_state,stratify=train_df['smoking_status'])

x_train = train_df.drop(['smoking_status'],axis=1)
y_train = train_df['smoking_status']

x_val = val_df.drop(['smoking_status'],axis=1)
y_val = val_df['smoking_status']

x_test = test_df.drop(['smoking_status'],axis=1)
y_test = test_df['smoking_status']

clf = clone(param_dict['boruta'])
clf.fit(x_train, y_train, eval_set=[(x_val, y_val)],eval_metric=['auc'], early_stopping_rounds=20, verbose=False)  # fit model
#calculate auc in test set
y_pred = clf.predict_proba(x_test)[:,1]  # predict

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

fpr_dic[name]=fpr
tpr_dic[name]=tpr
roc_auc_dic[name]=roc_auc 

for tag in ['ALPP','CXCL17', 'ACVRL1']:
    name = tag
    #select df_train where index is in eid_sex
    df_train_temp = df_train.copy()

    train_df, test_df = train_test_split(df_train_temp, test_size=0.3,random_state=random_state,stratify=df_train_temp['smoking_status'])
    train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=random_state,stratify=train_df['smoking_status'])

    #remove na in test_df
    test_df = test_df.dropna(subset=[tag])

    x_test = test_df[tag]
    y_test = test_df['smoking_status']

    fpr, tpr, _ = roc_curve(y_test, x_test)
    roc_auc = roc_auc_score(y_test, x_test)

    fpr_dic[name]=fpr
    tpr_dic[name]=tpr
    roc_auc_dic[name]=roc_auc 


plt.figure(figsize=(6, 4))
colors = cycle(['#14213d','#b3e2cd', '#fdcdac', '#cbd5e8'])
style = cycle(['-','--','--','--'])

sort_dic = {k: v for k, v in sorted(
    roc_auc_dic.items(), key=lambda item: item[1], reverse=True)}
for i, color in zip(sort_dic.keys(), colors):
    plt.plot(fpr_dic[i], tpr_dic[i], color=color, lw=2, alpha=.8, linestyle=next(style),  # plot results
             label=r'%s (AUC = %0.2f)' % (i, roc_auc_dic[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2, color='#b2182b',
         label='Chance', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('ROC in UKB test dataset',fontsize=16,fontweight='bold')
plt.legend(loc="lower right")

# %%


