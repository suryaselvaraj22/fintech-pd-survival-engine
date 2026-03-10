# Fintech Probability of Default (PD) Survival Engine

![AWS](https://img.shields.io/badge/Amazon_AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Survival Analysis](https://img.shields.io/badge/Machine_Learning-Cox_Proportional_Hazards-0194E2?style=for-the-badge)

## Executive Summary
This project implements an end-to-end Machine Learning pipeline to predict the **Probability of Default (PD) and Time-to-Event** for an unsecured personal loan portfolio using Survival Analysis.

By leveraging the **Cox Proportional Hazards Model** and **Amazon S3** for cloud storage, this solution moves beyond simple binary classification (Yes/No) to answer the critical financial question: *When* will a customer default? This enables the business to:
* **Forecast Expected Lifetime:** Dynamically predict the exact month a customer is expected to default based on their individual risk profile.
* **Optimize Intervention Timing:** Deploy targeted marketing or restructuring offers (e.g., refinancing) *before* the critical drop-off point.
* **Calculate Hazard Ratios:** Quantify the exact mathematical impact of financial friction (Debt-to-Income, Credit Score) on loan survivability.

## The Tech Stack
* **Core Logic:** Python, Pandas, Scikit-Learn
* **Modeling & Evaluation:** `lifelines` (CoxPHFitter, Concordance Index)
* **Cloud Infrastructure:** Amazon S3 (Data Lake Storage)
* **Ops & Deployment:** `boto3` (AWS SDK for Python)

## Key Results & Business Impact
The Cox Proportional Hazards model successfully separated routine payoffs from early defaults, evaluating performance via the out-of-sample Concordance Index (C-Index).

* **Concordance Index (C-Index):** Rigorously evaluated on a 20% unseen holdout set to prove real-world accuracy against unseen data.
* **Batch Scoring:** Successfully generated a 12-Month Survival Probability matrix for 10,000+ test records and seamlessly exported the predictions back to an S3 Data Lake for downstream business consumption.

### Extracted Business Intelligence (Hazard Ratios)
By extracting the Hazard Ratios, the model successfully isolated the mathematical drivers of default:
* **Debt-to-Income (Risk Accelerator):** A Hazard Ratio > 1.0 mathematically proved that high DTI heavily accelerates the risk of default, massively shrinking the Time-to-Event window.
* **Credit Score (Protective Shield):** A Hazard Ratio < 1.0 confirmed that higher credit scores protect against default, extending the survival timeline of the loan.
* **Loan Amount (Noise):** A Hazard Ratio of exactly 1.0 (with a high p-value) proved the model can successfully identify and ignore statistically insignificant variables.

## Solution Architecture

This repository is modularized into a 2-stage cloud pipeline:

### `01_pd_data_simulation.py`
Engineered a highly realistic synthetic dataset of 50,000 personal loans. Purposefully embedded hidden mathematical rules tying Debt-to-Income and Credit Scores to high hazard risks, simulating real-world credit behavior. Leveraged `boto3` to programmatically upload the raw data to a secure Amazon S3 bucket.

### `02_survival_modeling.py`
Pulled the raw data from Amazon S3 into Pandas and implemented an reproducible 80/20 Train/Test Split. Trained a `CoxPHFitter` to analyze the exact `months_to_event`. Extracted Hazard Ratios for executive reporting, batch-scored the entire test set for 12-month survival probabilities, and exported the final predictions back to S3.

## How to Run This Project
1. Clone this repository to your local machine using VS Code.
2. Install the required libraries (`pip install pandas numpy boto3 lifelines scikit-learn`).
3. Configure your AWS CLI with programmatic access keys (`aws configure`).
4. Ensure you have an active Amazon S3 bucket created and update the `bucket_name` variable.
5. Run the scripts sequentially (`01` through `02`).