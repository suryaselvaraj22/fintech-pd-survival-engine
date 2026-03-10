# 02_survival_modeling.py
# Objective: Download the synthetic credit data from Amazon S3 and train a 
# Cox Proportional Hazards model to predict the Time-to-Default.

import pandas as pd
import numpy as np
import boto3
from lifelines import CoxPHFitter   
from io import StringIO

print("Starting Survival Analysis Modeling Phase...")

# 1. Download the Data from Amazon S3
bucket_name = 'fintech-pd-models-nivi'
file_key = 'raw_data/simulated_credit_data.csv'  

print(f"Fetching data from s3://{bucket_name}/{file_key}...")

try:
    s3_client = boto3.client('s3')
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    body = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(body))
    print("✅ Successfully downloaded and loaded data into Pandas!")
except Exception as e:
    print(f"⚠️ Failed to fetch from S3. Loading local fallback file. Reason: {e}")
    df = pd.read_csv("simulated_credit_data.csv")
    print("✅ Successfully loaded local data into Pandas!")

# 2. Data Preparation
# We must drop the 'loan_id' because it is a primary key, not a predictive feature.
df_modeling = df.drop(columns=['loan_id'])

# 3. Fit the Cox Proportional Hazards Model
# The Cox model requires a specific format:
# - The "duration" column (Time-to-Event) is 'months_to_event'
# - The "event" column (Defaulted or Not) is 'default_event'
# - The rest of the columns are treated as covariates (features)
cph = CoxPHFitter()
cph.fit(
    df_modeling, 
    duration_col='months_to_event', 
    event_col='default_event'
    show_progress=True
)

# 4. Evaluate the Model (Concordance Index)
# C-Index is the Survival Analysis equivalent of the AUC-ROC score.
c_index = cph.concordance_index_
print("\n" + "=" * 50)
print(f"🏆 MODEL EVALUATION: Concordance Index (C-Index) = {c_index:.4f}")
print("=" * 50)
print("(A score above 0.85 is considered excellent in credit risk modeling.)\n")

# 5. Business Interpretation: Hazard Ratios
# Hazard Ratios > 1 mean the feature INCREASES the risk of default (accelerates it).
# Hazard Ratios < 1 mean the feature DECREASES the risk of default (protects against it).
print("Extracting Hazard Ratios (Feature Importances)...")
summary_df = cph.summary[['exp(coef)', 'p']]
summary_df = summary_df.rename(columns={'exp(coef)': 'hazard_ratio', 'p': 'p_value'})
summary_df = summary_df.sort_values(by='hazard_ratio', ascending=False)

print("\n--- HAZARD RATIOS ---")
print(summary_df)

print("\n✅ Survival Analysis Modeling Complete!")