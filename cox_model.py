##Association with diseases
#-------------------------
import pandas as pd  # import libraries
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt  # plot results
import numpy as np
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter
import math

#Read metabolic age
all_preds_df = pd.read_csv('file_name', index_col=0)  # read data
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
    cph.fit(  # fit model
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