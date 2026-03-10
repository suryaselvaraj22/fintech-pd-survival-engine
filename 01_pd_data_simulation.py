# 01_pd_data_simulation.py
# Objective: Generate synthetic credit loan data for a Probability of Default (PD) model.
# We will engineer features to simulate "Time-to-Default" for Survival Analysis,
# and upload the generated dataset directly to Amazon S3.

import pandas as pd
import numpy as np
import boto3
from io import StringIO
import os

print("Starting Credit Default Data Simulation for Survival Analysis...")

# 1. Generate Base Data (50,000 Loan Originations)
np.random.seed(42)  # For reproducibility
num_loans = 50000

# Engineer highly realistic credit features
data = {
    'loan_id': range(1, num_loans + 1),
    # FICO Scores range from 300 to 850, centered around 680
    "credit_score": np.clip(np.random.normal(680, 50, num_loans), 300, 850).astype(int),  
    # Debt-to-Income (DTI) ratio (typically between 10% and 50%)
    "dti_ratio": np.clip(np.random.normal(0.25, 0.10, num_loans), 0.05, 0.60),
    # Loan Amount ($5k to $40k)
    "loan_amount": np.clip(np.random.normal(15000, 8000, num_loans), 5000, 40000).astype(int),
     # Employment length in years (0 to 10+)
     "employment_length_yrs": np.random.randint(0, 11, num_loans)
}

df = pd.DataFrame(data)

# 2. Engineer the Hidden "Survival" Logic
# High DTI and Low Credit Score = High Risk = Faster Default
print("Calculating hidden default probabilities and time-to-event...")

# Base hazard score (higher score = more likely to default)
df['hazard_score'] = (df['dti_ratio'] * 2.5) + ((850 - df['credit_score']) / 200) + (df['employment_length_yrs'] * 0.05) + np.random.normal(0, 0.5, num_loans)

# Define the "Event" (1 = Defaulted, 0 = Paid Off / Censored)
# The top 15% highest hazard scores are marked as defaults
threshold = df['hazard_score'].quantile(0.85)
df['default_event'] = (df['hazard_score'] > threshold).astype(int)

# 3. Calculate "Time-to-Event" (Months)
# Survival Analysis REQUIRES two target columns: the Event, and the Time.
# If they default (1), when did it happen? (Usually early, months 2 - 18)
# If they didn't default (0), how long were they observed? (Months 24 - 36)

def calculate_time(row):
    if row['default_event'] == 1:
        # Defaults happen faster for people with very high hazard scores
        base_time = 24 - (row['hazard_score'] * 4)
        return max(1, base_time + np.random.normal(0, 2)) # Add noise, but ensure time is at least 1 month
    else:
        # Non-defaults survive for a long time (e.g., standard 36 month loan term)
        return int(np.clip(np.random.normal(30, 6), 12, 36))  
    
df['months_to_event'] = df.apply(calculate_time, axis=1)

# Clean up temporary columns
df = df.drop(columns=['hazard_score'])

print(f"✅ Generated {num_loans} loans. Overall Default Rate: {df['default_event'].mean():.1%}")

# 4. Upload to Amazon S3 using Boto3
bucket_name = 'fintech-pd-models-nivi'  
file_key = 'raw_data/simulated_credit_data.csv'

print(f"\nAttempting to upload data to S3 Bucket: {bucket_name}...")

try:
    # Convert DataFrame to CSV in-memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Initialize S3 client and upload
    s3_client = boto3.client('s3')

    # Upload to S3
    s3_client.put_object(
        Bucket=bucket_name, 
        Key=file_key, 
        Body=csv_buffer.getvalue()
    )
    
    print(f"✅ SUCCESS: Data successfully uploaded to s3://{bucket_name}/{file_key}")  

except Exception as e:
    print(f"⚠️ S3 Upload Skipped. Reason: {e}")
    print(f"Saving locally as 'simulated_credit_data.csv' instead.")
    df.to_csv("simulated_credit_data.csv", index=False)

print("\nData simulation and upload process completed.")
print("\nSample Data:")
print(df.head())

