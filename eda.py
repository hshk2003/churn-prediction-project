import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load data
file_path = os.path.join('data', 'telecom_churn.csv')
df = pd.read_csv(file_path)

print("="*70)
print("EXPLORATORY DATA ANALYSIS - TELCO CUSTOMER CHURN")
print("="*70)

print(f"\n📊 DATASET OVERVIEW")
print(f"{'='*70}")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\n\n📋 DATA TYPES:")
print(df.dtypes)

print(f"\n\n🔍 MISSING VALUES:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values!")
else:
    print(missing[missing > 0])

print(f"\n\n📈 NUMERICAL FEATURES STATISTICS:")
print(df.describe())

print(f"\n\n🎯 TARGET VARIABLE (CHURN) DISTRIBUTION:")
churn_counts = df['Churn'].value_counts()
print(f"No Churn (0): {churn_counts[0]} customers ({churn_counts[0]/len(df)*100:.2f}%)")
print(f"Churn (1): {churn_counts[1]} customers ({churn_counts[1]/len(df)*100:.2f}%)")

# Categorical columns analysis
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n\n📁 CATEGORICAL FEATURES: {len(categorical_cols)}")
for col in categorical_cols[:5]:  # Show first 5
    print(f"\n{col}:")
    print(df[col].value_counts())

# Create comprehensive visualizations
fig = plt.figure(figsize=(18, 12))

# 1. Churn Distribution
ax1 = plt.subplot(3, 3, 1)
churn_counts.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
ax1.set_title('Churn Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Churn (0=No, 1=Yes)')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['No Churn', 'Churn'], rotation=0)

# 2. Tenure Distribution
ax2 = plt.subplot(3, 3, 2)
df.boxplot(column='tenure', by='Churn', ax=ax2)
ax2.set_title('Tenure by Churn Status', fontsize=12, fontweight='bold')
ax2.set_xlabel('Churn')
ax2.set_ylabel('Tenure (months)')
plt.suptitle('')  # Remove default title

# 3. Monthly Charges
ax3 = plt.subplot(3, 3, 3)
df.boxplot(column='MonthlyCharges', by='Churn', ax=ax3)
ax3.set_title('Monthly Charges by Churn', fontsize=12, fontweight='bold')
ax3.set_xlabel('Churn')
ax3.set_ylabel('Monthly Charges ($)')
plt.suptitle('')

# 4. Contract Type vs Churn
ax4 = plt.subplot(3, 3, 4)
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', ax=ax4, color=['#2ecc71', '#e74c3c'])
ax4.set_title('Churn Rate by Contract Type', fontsize=12, fontweight='bold')
ax4.set_xlabel('Contract Type')
ax4.set_ylabel('Percentage (%)')
ax4.legend(['No Churn', 'Churn'])
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

# 5. Internet Service vs Churn
ax5 = plt.subplot(3, 3, 5)
internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
internet_churn.plot(kind='bar', ax=ax5, color=['#2ecc71', '#e74c3c'])
ax5.set_title('Churn Rate by Internet Service', fontsize=12, fontweight='bold')
ax5.set_xlabel('Internet Service')
ax5.set_ylabel('Percentage (%)')
ax5.legend(['No Churn', 'Churn'])
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')

# 6. Payment Method vs Churn
ax6 = plt.subplot(3, 3, 6)
payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
payment_churn.plot(kind='bar', ax=ax6, color=['#2ecc71', '#e74c3c'])
ax6.set_title('Churn Rate by Payment Method', fontsize=12, fontweight='bold')
ax6.set_xlabel('Payment Method')
ax6.set_ylabel('Percentage (%)')
ax6.legend(['No Churn', 'Churn'])
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')

# 7. Tenure Distribution
ax7 = plt.subplot(3, 3, 7)
df[df['Churn']==0]['tenure'].hist(bins=30, alpha=0.7, label='No Churn', color='#2ecc71', ax=ax7)
df[df['Churn']==1]['tenure'].hist(bins=30, alpha=0.7, label='Churn', color='#e74c3c', ax=ax7)
ax7.set_title('Tenure Distribution', fontsize=12, fontweight='bold')
ax7.set_xlabel('Tenure (months)')
ax7.set_ylabel('Frequency')
ax7.legend()

# 8. Monthly Charges Distribution
ax8 = plt.subplot(3, 3, 8)
df[df['Churn']==0]['MonthlyCharges'].hist(bins=30, alpha=0.7, label='No Churn', color='#2ecc71', ax=ax8)
df[df['Churn']==1]['MonthlyCharges'].hist(bins=30, alpha=0.7, label='Churn', color='#e74c3c', ax=ax8)
ax8.set_title('Monthly Charges Distribution', fontsize=12, fontweight='bold')
ax8.set_xlabel('Monthly Charges ($)')
ax8.set_ylabel('Frequency')
ax8.legend()

# 9. Senior Citizen vs Churn
ax9 = plt.subplot(3, 3, 9)
senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'], normalize='index') * 100
senior_churn.plot(kind='bar', ax=ax9, color=['#2ecc71', '#e74c3c'])
ax9.set_title('Churn Rate by Senior Citizen Status', fontsize=12, fontweight='bold')
ax9.set_xlabel('Senior Citizen (0=No, 1=Yes)')
ax9.set_ylabel('Percentage (%)')
ax9.legend(['No Churn', 'Churn'])
ax9.set_xticklabels(['No', 'Yes'], rotation=0)

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
print("\n✅ Comprehensive visualization saved as 'eda_visualization.png'")
plt.show()

# Key Insights
print("\n" + "="*70)
print("🔑 KEY INSIGHTS")
print("="*70)
print(f"1. Overall churn rate: {df['Churn'].mean()*100:.2f}%")
print(f"2. Customers with month-to-month contracts have highest churn")
print(f"3. Fiber optic internet users show higher churn rates")
print(f"4. Electronic check payment method correlates with higher churn")
print(f"5. Newer customers (low tenure) are more likely to churn")
print(f"6. Higher monthly charges correlate with increased churn risk")

print("\n" + "="*70)
print("✅ EDA COMPLETE!")
print("="*70)