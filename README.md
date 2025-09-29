
---

# Proteomic Smoking INdex (pSIN) Project

This repository contains code to reproduce and extend the analyses in:
“Proteomic signatures of smoking and their associations with risk of incident diseases and mortality in diverse populations” :contentReference[oaicite:0]{index=0}

Overview
--------
We developed a plasma‐proteome–based Smoking INdex (pSIN) using machine learning in UK Biobank (UKB; n≈44k) and validated it in China Kadoorie Biobank (CKB; n≈4k).  We then studied genetic/exposome determinants and associations of pSIN with morbidity, mortality, and biomarker profiles.

Directory structure
-------------------
- `annotate_proteins.py`  
- `model_training.py`  
- `predictability_by_haematology.py`  
- `CKB_data_preparation.py`  
- `external_validation.py`  
- `exposome_and_contribution_to_pSIN.py`  
- `pSIN.py`  
- `cumulative_incidence_plot.py`  
- `UKB_data_preparation.py`  
- `UKB_by_sex_and_undersampling.py`  
- `Check_age_sex.py`  
- `pSINvsProtein.py`  
- `cox_model.py`  
- `linear_associations.py`  


Requirements
------------
- Python 3.8+  
# Core data & ML libraries
pip install pandas numpy scikit-learn lightgbm optuna 

# HTTP & I/O
pip install requests tqdm

# Statistical & modeling
pip install statsmodels lifelines shap

# Plotting
pip install matplotlib seaborn adjustText

# Utilities
pip install joblib imbalanced-learn shaphypetune

````

## File descriptions & usage

### 1. `UKB_data_preparation.py`

Preprocesses UKB Olink proteomic NPX data: regresses out age and normalizes per batch, filters proteins with >20% missing, and exports a Feather file for downstream analysis.

**Inputs:**

* Raw Olink CSV (`olink_data.csv`)
* UKB covariate Feather (`ukb_full.feather`)

**Outputs:**

* `ukb_proteome_processed.feather`

---

### 2. `CKB_data_preparation.py`

Analogous preprocessing for CKB proteomic data: merges multiple batches, regresses age, normalizes, filters and aligns proteins with UKB panel.

**Inputs:**

* CKB proteomic CSVs (`ckb_batch*.csv`)
* CKB metadata (`ckb_metadata.csv`)

**Outputs:**

* `ckb_proteome_processed.feather`

---

### 3. `model_training.py`

Trains and tunes a LightGBM classifier to discriminate current vs. never smokers using UKB proteome.

* **Stage 1:** Optuna hyperparameter tuning (`optuna_lgbm`)
* **Stage 2:** Boruta‐SHAP for feature selection
* **Stage 3:** Final model fitting on selected 51 proteins → saves `param_dict_clf.pkl` and `boruta_model.pkl`.


---

### 4. `pSIN.py`

Calculates proteomic Smoking INdex (pSIN) scores:

* Loads trained model & Boruta selector
* Computes raw scores on UKB, CKB, and self‐reported “previous”/“never‐H” groups
* Generates SHAP summary and ROC plots (saved to `plots/`).


---

### 5. `external_validation.py`

Applies the trained classifier to CKB data to assess external performance via bootstrapped AUC, F1, precision, balanced accuracy.

---

### 6. `predictability_by_haematology.py`

Assesses to what extent standard haematology indices predict smoking status (vs. pSIN) via 5-fold CV and bootstrap metrics.

---

### 7. `UKB_by_sex_and_undersampling.py`

Re‐evaluates classifier performance in UKB stratified by sex, and underbalanced test splits (1:1 current\:never) via bootstrapping.

---

### 8. `Check_age_sex.py`

Quantifies how each protein alone predicts age (R²) and sex (AUC) using single‐feature LightGBM models in parallel.

---

### 9. `pSINvsProtein.py`

Compares multi‐protein pSIN vs. top three proteins (ALPP, CXCL17, ACVRL1) on ROC curves in UKB test set.

---

### 10. `exposome_and_contribution_to_pSIN.py`

Quantifies variance in pSIN explained by genetics (GWAS SNPs), smoking history, exposome (XWAS + behavioural), and clinical biomarkers using sequential LightGBM regression models.

---

### 11. `linear_associations.py`

Tests associations between pSIN and baseline clinical biomarkers, blood counts, and phenotypes via multivariable linear models (OLS) adjusting for sociodemographics.

---

### 12. `cox_model.py`

Fits Cox proportional hazards models to relate pSIN (per‐SD) to incident risk of 27 major diseases and all‐cause mortality under three covariate sets:

* Model 1: pSIN only
* Model 2: + sociodemographics/lifestyle
* Model 3: + pack‐years

Generates forest plot of HRs & 95% CIs.

---

### 13. `cumulative_incidence_plot.py`

Plots age‐based cumulative incidence curves (Kaplan–Meier) by pSIN quartiles in UKB overall, in current smokers, and in previous smokers (“recovered” vs “non‐recovered”) for top 18 outcomes.

---
### Data Access Statement
UK Biobank data are available through a procedure described at: https://www.ukbiobank.ac.uk/enable-your-research. 
The China Kadoorie Biobank (CKB) is a global resource for the investigation of lifestyle, environmental, blood biochemical and genetic factors as determinants of common diseases. The CKB study group is committed to making the cohort data available to the scientific community in China, the UK, and worldwide to advance knowledge about the causes, prevention and treatment of disease. For detailed information on what data is currently available to open access users and how to apply for it, please visit: https://www.ckbiobank.org/data-access. A research proposal will be requested to ensure that any analysis is performed by bona fide researchers. Researchers who are interested in obtaining additional information or data that underlines this paper should contact ckbaccess@ndph.ox.ac.uk. For any data that is not currently available to open access, researchers may need to develop formal collaboration with the CKB study group. 

---
This is for academic use only. No commercial use is allowed.
Questions or issues? Please open an issue or contact **[sihao.xiao@bnc.ox.ac.uk](mailto:sihao.xiao@bnc.ox.ac.uk)**.

