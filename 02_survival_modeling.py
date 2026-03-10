# 02_survival_modeling.py
# Objective: Download the synthetic credit data from Amazon S3 and train a 
# Cox Proportional Hazards model to predict the Time-to-Default.

import pandas as pd
import numpy as np
import boto3
from lifelines import CoxPHFitter   
from io import StringIO
from sklearn.model_selection import train_test_split    

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

# 2. Data Preparation & Train/Test Split
# We must drop the 'loan_id' because it is a primary key, not a predictive feature.
df_modeling = df.drop(columns=['loan_id'])

# Perform the 80/20 Split to prevent data leakage
print("\nSplitting data into 80% Training and 20% Testing sets...")
train_data, test_data = train_test_split(df_modeling, test_size=0.2, random_state=42)
print(f"Training Set: {len(train_data)} records")
print(f"Testing Set: {len(test_data)} records")

# 3. Initialize and Train the Cox Proportional Hazards (CPH) Model
# The Cox model requires a specific format:
# - The "duration" column (Time-to-Event) is 'months_to_event'
# - The "event" column (Defaulted or Not) is 'default_event'
# - The rest of the columns are treated as covariates (features)
print("\nTraining the Cox Proportional Hazards Model on the 80% Training Set...")
cph = CoxPHFitter()

# We fit the model ONLY on train_data
cph.fit(
    train_data, 
    duration_col='months_to_event', 
    event_col='default_event',
    show_progress=True
)

# 4. Evaluate the Model (Concordance Index on UNSEEN data)
# C-Index is the Survival Analysis equivalent of the AUC-ROC score.
# By scoring on test_data, we prove the model's true real-world accuracy.
c_index_test = cph.score(test_data, scoring_method="concordance_index")
print("\n" + "=" * 50)
print(f"🏆 MODEL EVALUATION: Concordance Index (C-Index) = {c_index_test:.4f}")
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

# 6. Making Predictions (The "Whether" and "When")
print("\n" + "=" * 50)
print("🔮 PREDICTING THE FUTURE: WHEN WILL THEY DEFAULT?")
print("=" * 50)

# Grab 3 sample customers from the UNSEEN Testing Set
sample_customers = test_data.head(3)

# A) Predicting "When" (Expected Lifetime / Time-to-Event)
expected_months = cph.predict_expectation(sample_customers)

# B) Predicting "Whether" (Probability of surviving past a specific point)
# This generates a timeline matrix for each customer
survival_curves = cph.predict_survival_function(sample_customers)

for i in range(len(sample_customers)):
    # We look exactly at month 12 in the survival curve to get their 1-year safety probability
    prob_12m = survival_curves.iloc[:, i].get(12, "N/A")  # Row 12 corresponds to month 12
    if isinstance(prob_12m, float):
        prob_12m = f"{prob_12m:.1%}"
    print(f"Customer {i+1} (from test set):")
    print(f"  -> Expected Time to Default (The 'When'):    {expected_months.iloc[i]:.1f} months")
    print(f"  -> 12-Month Survival Prob (The 'Whether'):   {prob_12m}")
    print("-" * 40)

# 7. Productionizing: Batch Scoring & S3 Export
print("\n" + "=" * 50)
print("🚀 PRODUCTION DEPLOYMENT: BATCH SCORING TO S3")
print("=" * 50)
print(f"Scoring all {len(test_data)} customers in the testing set...")

# Predict for the entire test dataset
predicted_expected_months = cph.predict_expectation(test_data)
# Efficiently predict survival probability specifically at month 12 for all customers
predicted_survival_12m = cph.predict_survival_function(test_data, times=[12]).iloc[0, :]

# Attach predictions to our dataframe
predictions_df = test_data.copy()
predictions_df['predicted_months_to_default'] = predicted_expected_months.round(1)
predictions_df['predicted_survival_prob_12m'] = predicted_survival_12m.round(4)

# Export back to S3
output_file_key = 'predictions/scored_customers.csv'
print(f"Uploading batch predictions to s3://{bucket_name}/{output_file_key}...")

try:
    csv_buffer_out = StringIO()
    predictions_df.to_csv(csv_buffer_out, index=False)
    
    s3_client.put_object(
        Bucket=bucket_name, 
        Key=output_file_key, 
        Body=csv_buffer_out.getvalue()
    )
    print("✅ SUCCESS: Predictions successfully uploaded to AWS S3!")
except Exception as e:
    print(f"⚠️ S3 Upload Skipped. Reason: {e}")
    print("Saving predictions locally as 'scored_customers.csv' instead.")
    predictions_df.to_csv("scored_customers.csv", index=False)

print("\n✅ Survival Analysis Modeling Complete!")