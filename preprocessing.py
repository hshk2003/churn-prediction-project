import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

print("="*70)
print("DATA PREPROCESSING - TELCO CUSTOMER CHURN")
print("="*70)

# Load data
file_path = os.path.join('data', 'telecom_churn.csv')
df = pd.read_csv(file_path)
print(f"\n📊 Original dataset shape: {df.shape}")

# Check for any remaining missing values
print(f"\n🔍 Checking for missing values...")
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Missing values found:")
    print(missing[missing > 0])
else:
    print("✅ No missing values")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"\n📋 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n📁 Categorical features ({len(categorical_cols)}):")
for col in categorical_cols:
    print(f"   • {col}: {X[col].nunique()} unique values")

print(f"\n🔢 Numerical features ({len(numerical_cols)}):")
for col in numerical_cols:
    print(f"   • {col}: range [{X[col].min():.2f}, {X[col].max():.2f}]")

# Encode categorical variables
print(f"\n🔄 Encoding categorical variables...")
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"   ✓ {col}: {list(le.classes_)[:3]}{'...' if len(le.classes_) > 3 else ''}")

# Save encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print(f"\n💾 Label encoders saved to 'label_encoders.pkl'")

# Split data (80% train, 20% test)
print(f"\n✂️  Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain same churn ratio in both sets
)

print(f"\n📦 Dataset splits:")
print(f"   • Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"   • Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\n   • Train churn rate: {y_train.mean()*100:.2f}%")
print(f"   • Test churn rate: {y_test.mean()*100:.2f}%")

# Scale numerical features (important for model performance!)
print(f"\n⚖️  Scaling numerical features...")
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"   ✓ Scaled {len(numerical_cols)} numerical features")

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(numerical_cols, 'numerical_cols.pkl')  # Save column names too
print(f"   💾 Scaler saved to 'scaler.pkl'")

# Save processed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print(f"\n💾 Preprocessed data saved:")
print(f"   • X_train.csv")
print(f"   • X_test.csv")
print(f"   • y_train.csv")
print(f"   • y_test.csv")

print("\n" + "="*70)
print("✅ PREPROCESSING COMPLETE!")
print("="*70)
print("\nYou're ready to train models! Run: python train_model.py")