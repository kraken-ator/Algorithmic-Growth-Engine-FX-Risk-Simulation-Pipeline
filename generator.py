import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Initializing Algorithmic Growth Engine...")

#PHASE 1: SYNTHESIZE THE B2B MARKET
np.random.seed(42)
num_records = 5000

industries = ['IT Service Agency', 'SaaS Startup', 'Freelance Developer', 'D2C E-commerce']
gateways = ['Traditional Bank', 'PayPal', 'Payoneer', 'Stripe']

data = {
    'Company_ID': [f'IND-{i+1000}' for i in range(num_records)],
    'Industry_Niche': np.random.choice(industries, num_records, p=[0.4, 0.2, 0.3, 0.1]),
    'Current_Gateway': np.random.choice(gateways, num_records, p=[0.4, 0.3, 0.2, 0.1]),
}
df = pd.DataFrame(data)

def generate_volume(industry):
    if industry == 'IT Service Agency': return np.random.normal(35000, 10000)
    elif industry == 'SaaS Startup': return np.random.normal(20000, 6000)
    elif industry == 'Freelance Developer': return np.random.normal(5000, 2000)
    else: return np.random.normal(15000, 5000)

df['Monthly_Volume_USD'] = df['Industry_Niche'].apply(generate_volume).clip(lower=1000).round(2)

def assign_markup(gateway):
    if gateway == 'Traditional Bank': return np.random.uniform(0.025, 0.05) 
    elif gateway == 'PayPal': return np.random.uniform(0.03, 0.045)
    elif gateway == 'Payoneer': return np.random.uniform(0.015, 0.03)
    else: return np.random.uniform(0.015, 0.025)

df['Base_Forex_Markup'] = df['Current_Gateway'].apply(assign_markup)

# PHASE 2: STOCHASTIC FX RISK SIMULATION (Monte Carlo)
print("Running Monte Carlo FX Simulations...")
annual_volatility = 0.06 
monthly_volatility = annual_volatility / np.sqrt(12)

fx_shocks = np.random.normal(0, monthly_volatility, (num_records, 12))
worst_case_shock = np.percentile(fx_shocks, 95, axis=1)

df['Simulated_FX_Risk_Multiplier'] = 1 + worst_case_shock
df['Effective_Monthly_Fee_Lost_USD'] = (df['Monthly_Volume_USD'] * df['Base_Forex_Markup'] * df['Simulated_FX_Risk_Multiplier']).round(2)


# PHASE 3: ML PROPENSITY-TO-CONVERT MODEL
print("Training Propensity-to-Convert ML Model...")

loss_ratio = df['Effective_Monthly_Fee_Lost_USD'] / df['Monthly_Volume_USD']
conversion_logic = (loss_ratio > 0.025) & (df['Monthly_Volume_USD'] > 10000)
df['Historical_Conversion_Status'] = np.where(conversion_logic, 
                                              np.random.choice([1, 0], num_records, p=[0.8, 0.2]), 
                                              np.random.choice([1, 0], num_records, p=[0.1, 0.9]))

features = ['Monthly_Volume_USD', 'Effective_Monthly_Fee_Lost_USD', 'Base_Forex_Markup']
X = df[features]
y = df['Historical_Conversion_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(X)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)

df['Probability_to_Convert'] = rf_model.predict_proba(X_scaled)[:, 1].round(4)


# PHASE 4: DYNAMIC CAC OPTIMIZATION 
print("Calculating Dynamic LTV & CAC Optimizations...")

df['Est_Skydo_Monthly_Rev'] = ((df['Monthly_Volume_USD'] / 2000) * 19)

df['Projected_24M_LTV_USD'] = (df['Est_Skydo_Monthly_Rev'] * 24 * 0.95).round(2)

df['Expected_Value_USD'] = (df['Projected_24M_LTV_USD'] * df['Probability_to_Convert']).round(2)

df['Optimized_Target_CAC_USD'] = (df['Expected_Value_USD'] * 0.30).round(2)

conditions = [
    (df['Probability_to_Convert'] >= 0.75),
    (df['Probability_to_Convert'] >= 0.40),
    (df['Probability_to_Convert'] < 0.40)
]
choices = ['High Propensity (>75%)', 'Medium Propensity (40-75%)', 'Low Propensity (<40%)']
df['GTM_Target_Priority'] = np.select(conditions, choices, default='Unknown')

# PHASE 5: EXPORT FOR DASHBOARD
columns_to_export = [
    'Company_ID', 'Industry_Niche', 'Current_Gateway', 'Monthly_Volume_USD', 
    'Effective_Monthly_Fee_Lost_USD', 'Projected_24M_LTV_USD', 
    'Probability_to_Convert', 'Optimized_Target_CAC_USD', 'GTM_Target_Priority'
]
final_df = df[columns_to_export]

csv_filename = "skydo_advanced_growth_engine.csv"
final_df.to_csv(csv_filename, index=False)

print(f"✅ Heavyweight Pipeline Complete! Data saved as {csv_filename}")
print("-" * 30)
print("Target Priority Breakdown:")
print(final_df['GTM_Target_Priority'].value_counts())
