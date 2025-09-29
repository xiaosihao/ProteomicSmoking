## Cumulative incidence plot
#--------------------------------
import pandas as pd  # import libraries
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt  # plot results
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

all_ncd = pd.read_feather("file_name")  # read data
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
    kmf.fit(T[ix], E[ix], label='Top 25%')  # fit model
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

cum_den_df.to_csv('file_name', index=False)  # save to csv

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