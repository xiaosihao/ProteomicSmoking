# %%
import pandas as pd  # import libraries
from tqdm import tqdm

# %%
#read csv file
olink_data = pd.read_csv("PathName/FileName",index_col=0)  # read data
olink_data.set_index('eid',inplace=True)

#read ukb file
ukb = pd.read_feather("PathName/FileName")
ukb = ukb.set_index('eid')



# %%
olink_data['DDC'].median()

# %%
olink_data

# %%
#left join ukb and olink_data on index 
df = olink_data.iloc[:,:-2].join(ukb[['sex','recruitment_age']], how='inner')
#replace Male with 0 and Female with 1 in sex column
df['sex'] = df['sex'].replace({'Male': 0, 'Female': 1})

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
for i in tqdm(olink_data.iloc[:,:-2].columns):
    new_col = normalize_median_center(regress_age(df[i],df['recruitment_age']))
    new_columns.append(pd.Series(new_col, name=i))
new_df = pd.concat(new_columns,axis=1)


# %%
#add back batch information
new_df = new_df.join(olink_data.iloc[:,-2:])
#only random selected samples
new_df = new_df[~new_df['olink_batch'].isin([0,7])]
#remove last 2 columns (olink_batch and olink_plate)
new_df = new_df.iloc[:,:-2]
#remove columns with more than 20% of missing values
new_df = new_df[new_df.isna().sum(axis=1) <= 0.20*new_df.shape[1]]
#remove proteins with mroe than 20% of missing values
new_df = new_df[[i for i in new_df.columns if i not in ['NPM1', 'PCOLCE', 'CTSS', 'GLIPR1']]]
#remove proteins in UKB but not in CKB
new_df = new_df[[i for i in new_df.columns if i not in ['HLA_A','ERVV_1']]]
#remove proteins in CKB but not in UKB
new_df = new_df[[i for i in new_df.columns if i not in ['CD97','FGFR1OP','LRMP','CASC4','DARS','HARS','WISP2','FOPNL','WISP1']]]


# %%
#reset index
new_df.reset_index(inplace=True)
#save as feather file
new_df.to_feather("PathName/FileName")

# %%


