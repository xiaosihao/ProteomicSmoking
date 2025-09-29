# %%
import pandas as pd  # import libraries
from tqdm import tqdm

# %%
ckb_data1 = pd.read_csv("PathName/FileName",index_col=0)  # read data
ckb_data2 = pd.read_csv("PathName/FileName",index_col=0)
ckb_data3 = pd.read_csv("PathName/FileName",index_col=0)
ckb_data4 = pd.read_csv("PathName/FileName",index_col=0)

# %%
#merge all data based on csid if columns are the same, only keep the first one
ckb_data = pd.concat([ckb_data1, ckb_data2, ckb_data3, ckb_data4], axis=1)
ckb_data = ckb_data.loc[:,~ckb_data.columns.duplicated()]

#remove ol_ prefix
ckb_data.columns = [i.replace('ol_','') for i in ckb_data.columns]
#change to upper case
ckb_data.columns = [i.upper() for i in ckb_data.columns]

#remove columns if they are empty
for i in tqdm(ckb_data.columns):
    if ckb_data[i].isnull().sum() == ckb_data.shape[0]:
        ckb_data = ckb_data.drop(columns=[i])
        print(i)


# %%
ckb = pd.read_csv("PathName/FileName",index_col=0)
#only those in ckb_data index
ckb = ckb.loc[ckb_data.index,:]
#add to ckb_data
ckb_data = ckb_data.join(ckb[['is_female','age_at_study_date_x100']],how='left')
#rename age
ckb_data.rename(columns={'age_at_study_date_x100':'recruitment_age','is_female':'sex'},inplace=True)
#format age
ckb_data['recruitment_age'] = ckb_data['recruitment_age']/100
#round age
ckb_data['recruitment_age'] = ckb_data['recruitment_age'].round(0)


# %%
def regress_age(series,age):
    import statsmodels.api as sm
    #take original index
    orig_idx = series.index
    #create a df and remove na
    df = pd.DataFrame({'y':series,'x':age}).dropna()
    #run regression
    model = sm.OLS(df['y'],sm.add_constant(df['x'])).fit()  # fit model
    #get residuals
    res = model.resid
    #add residuals to mean
    mean = df['y'].mean()
    new_value = res + mean
    #create a new series with original index and those removed as na
    new_series = pd.Series(new_value,index=orig_idx)
    return new_series

def normalize_median_center(series):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    scaler = MinMaxScaler()
    #fit and transform
    new_series = scaler.fit_transform(series.values.reshape(-1, 1))
    #put back to series
    new_series = pd.Series(new_series.reshape(-1),index=series.index)
    #calculate median
    median = new_series.median()
    #center median
    new_series = new_series - median
    return new_series



# %%
new_df = pd.DataFrame()
new_columns = []
for i in tqdm(ckb_data.iloc[:,:-2].columns):
    new_col = normalize_median_center(regress_age(ckb_data[i],ckb_data['recruitment_age']))
    new_columns.append(pd.Series(new_col, name=i))
new_df = pd.concat(new_columns,axis=1)


# %%
#remove proteins with mroe than 20% of missing values
new_df = new_df[[i for i in new_df.columns if i not in ['NPM1', 'PCOLCE', 'CTSS', 'GLIPR1']]]
#remove proteins in UKB but not in CKB
new_df = new_df[[i for i in new_df.columns if i not in ['HLA_A','ERVV_1']]]
#remove proteins in CKB but not in UKB
new_df = new_df[[i for i in new_df.columns if i not in ['CD97','FGFR1OP','LRMP','CASC4','DARS','HARS','WISP2','FOPNL','WISP1']]]
#change names
new_df = new_df.rename(columns={'NTPROBNP':'NTproBNP','C19ORF12':'C19orf12','C2ORF69':'C2orf69','C7ORF50':'C7orf50','C9ORF40':'C9orf40'})

# %%
#read csv file
olink_data = pd.read_feather("PathName/FileName")
olink_data = olink_data.set_index('eid')

# %%
[i for i in new_df.columns if i not in olink_data.columns]

# %%
#reset index
new_df.reset_index(inplace=True)
#save as feather
new_df.to_feather("PathName/FileName")
