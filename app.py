from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
numerical_cols = joblib.load('numerical_cols.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Create DataFrame with all expected features
        # This ensures correct column order matching training data
        feature_order = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
        
        # Fill in missing features with defaults if not in form
        defaults = {
            'gender': 'Male',
            'SeniorCitizen': '0',
            'Partner': 'No',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No'
        }
        
        for feature in feature_order:
            if feature not in data:
                data[feature] = defaults.get(feature, 'No')
        
        input_df = pd.DataFrame([data])
        
        # Reorder columns to match training
        input_df = input_df[feature_order]
        
        # Convert numerical fields
        for field in numerical_cols:
            input_df[field] = pd.to_numeric(input_df[field])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Scale numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        result = {
            'prediction': 'Will Churn' if prediction == 1 else 'Will Not Churn',
            'churn_probability': float(probability[1] * 100),
            'retention_probability': float(probability[0] * 100)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)