# %%
import pandas as pd  # import libraries


# %%
#read csv file
olink_data = pd.read_feather("PathName/FileName")  # read data
olink_data = olink_data.set_index('eid')

# ukb = pd.read_feather("PathName/FileName")
ukb = pd.read_feather("PathName/FileName")
ukb = ukb.set_index('eid')

#remove overlapping columns
ukb = ukb.drop(columns = [i for i in ukb.columns if i in olink_data.columns])

df = ukb[['recruitment_age','sex']].join(olink_data,how='inner')


# %%
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split  # split data
from sklearn.metrics import r2_score, roc_auc_score
from joblib import Parallel, delayed

# Assuming df is your DataFrame
protein_cols = df.columns[2:]  # Protein columns

# Function to train model and compute R² and AUC for a single protein
def train_and_evaluate(protein):
    # Predict age using the protein
    y_age = df['recruitment_age']
    y_sex = df['sex']
    
    # Remove rows where protein values are NaN
    valid_idx = df[protein].notna()
    X_clean = df.loc[valid_idx, [protein]]
    y_age_clean = y_age.loc[valid_idx]
    y_sex_clean = y_sex.loc[valid_idx]

    # If not enough data after dropping NaNs, return None
    if len(y_age_clean) < 10 or len(y_sex_clean) < 10:  # Set a minimum threshold (adjust if needed)
        return {'Protein': protein, 'R2_age': None, 'AUC_sex': None}
    
    # Split data into training and testing sets
    X_train, X_test, y_age_train, y_age_test, y_sex_train, y_sex_test = train_test_split(
        X_clean, y_age_clean, y_sex_clean, test_size=0.3, random_state=1996
    )
    
    # Train LGBM Regressor for age prediction
    model_age = lgb.LGBMRegressor()
    model_age.fit(X_train, y_age_train)  # fit model
    
    # Train LGBM Classifier for sex prediction
    model_sex = lgb.LGBMClassifier()
    model_sex.fit(X_train, y_sex_train)
    
    # Get predictions and R² score for age
    y_age_pred = model_age.predict(X_test)  # predict
    r2_age = r2_score(y_age_test, y_age_pred)
    
    # Get predictions and AUC score for sex
    y_sex_pred_proba = model_sex.predict_proba(X_test)[:, 1]
    auc_sex = roc_auc_score(y_sex_test, y_sex_pred_proba)
    
    return {'Protein': protein, 'R2_age': r2_age, 'AUC_sex': auc_sex}

# Run in parallel using joblib
num_cores = -1  # Use all available CPU cores
results = Parallel(n_jobs=num_cores)(
    delayed(train_and_evaluate)(protein) for protein in protein_cols
)

# Convert results to DataFrame
results_df = pd.DataFrame(results)


# %%
# print mean and standard deviation of R² and AUC
print(results_df['R2_age'].mean(), results_df['R2_age'].std())

print(results_df['AUC_sex'].mean(), results_df['AUC_sex'].std())


