#linear assocation models
#-------------------------

import pandas as pd  # import libraries
import matplotlib.pyplot as plt  # plot results
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

#Read metabolic age
all_preds_df = pd.read_csv('file_name',index_col=0)  # read data
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
        model = sm.formula.ols(formula=formula, data=temp_df).fit()  # fit model
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