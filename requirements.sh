#!/usr/bin/env bash
python3 -m pip install --upgrade pip

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
