import pandas as pd
import numpy as np
import random

def generate_logistics_data(num_samples=1000):
    np.random.seed(42) # Ensure reproducibility

    data = {
        'company_id': [f"CUST_{i:04d}" for i in range(num_samples)],
        'industry': np.random.choice(['Retail', 'Automotive', 'Pharma', 'Electronics', 'Perishable'], num_samples),
        'monthly_spend_usd': np.random.normal(15000, 5000, num_samples).astype(int),
        'shipments_per_month': np.random.poisson(20, num_samples),
        'avg_delay_hours': np.random.normal(12, 5, num_samples), # Avg delay in hours
        'support_tickets_last_90d': np.random.poisson(2, num_samples),
        'contract_length_months': np.random.choice([12, 24, 36], num_samples, p=[0.6, 0.3, 0.1]),
        'has_api_integration': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 1 = Yes, 0 = No
    }
    
    df = pd.DataFrame(data)

    # --- Inject Business Logic for Churn (The "Signal" for the Model) ---
    # We want the model to learn that High Delays + High Tickets = Churn
    
    # Base churn probability
    df['churn_probability'] = 0.15 
    
    # 1. Retail clients are flighty (higher churn)
    df.loc[df['industry'] == 'Retail', 'churn_probability'] += 0.10
    
    # 2. API Integration creates "Stickiness" (lower churn)
    df.loc[df['has_api_integration'] == 1, 'churn_probability'] -= 0.10
    
    # 3. High delays cause churn (Strong signal)
    df.loc[df['avg_delay_hours'] > 18, 'churn_probability'] += 0.30
    
    # 4. Bad support experience causes churn
    df.loc[df['support_tickets_last_90d'] > 4, 'churn_probability'] += 0.25

    # Clip probability between 0 and 1
    df['churn_probability'] = df['churn_probability'].clip(0, 1)

    # Generate the actual target variable (1 = Churn, 0 = Stay) based on probability
    df['churned'] = np.random.binomial(1, df['churn_probability'])

    # Cleanup: Remove the probability column (the model shouldn't see this!)
    df_final = df.drop(columns=['churn_probability'])
    
    # Make sure we don't have negative numbers for spend/delays
    df_final['monthly_spend_usd'] = df_final['monthly_spend_usd'].abs()
    df_final['avg_delay_hours'] = df_final['avg_delay_hours'].abs()

    print(f"Generated {num_samples} records.")
    print(f"Overall Churn Rate: {df_final['churned'].mean():.2%}")
    
    return df_final

if __name__ == "__main__":
    df = generate_logistics_data()
    df.to_csv('logistics_churn_data.csv', index=False)
    print("File saved: logistics_churn_data.csv")