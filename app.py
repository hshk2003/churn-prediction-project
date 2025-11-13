from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
numerical_cols = joblib.load('numerical_cols.pkl')

# Define the minimal feature set we're using
REQUIRED_FEATURES = [
    'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges',
    'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity'
]

# Complete feature list that model expects (in order)
ALL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form (only 8 fields)
        user_data = request.form.to_dict()
        
        print("📥 Received data:", user_data)
        
        # Create full feature dictionary with smart defaults
        full_data = {
            # Demographics - safe defaults
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            
            # From user input
            'tenure': user_data.get('tenure', 12),
            
            # Phone services - safe defaults
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            
            # From user input
            'InternetService': user_data.get('InternetService', 'Fiber optic'),
            'OnlineSecurity': user_data.get('OnlineSecurity', 'No'),
            
            # Additional services - safe defaults
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            
            # From user input
            'TechSupport': user_data.get('TechSupport', 'No'),
            
            # Streaming - safe defaults
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            
            # From user input
            'Contract': user_data.get('Contract', 'Month-to-month'),
            
            # Billing - safe default
            'PaperlessBilling': 'Yes',
            
            # From user input
            'PaymentMethod': user_data.get('PaymentMethod', 'Electronic check'),
            'MonthlyCharges': user_data.get('MonthlyCharges', 70),
            'TotalCharges': user_data.get('TotalCharges', 840)
        }
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([full_data], columns=ALL_FEATURES)
        
        # Convert numerical columns
        numerical_fields = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        for field in numerical_fields:
            input_df[field] = pd.to_numeric(input_df[field], errors='coerce').fillna(0)
        
        print("📊 DataFrame before encoding:\n", input_df)
        
        # Encode categorical variables
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except Exception as e:
                    print(f"⚠️ Warning encoding {col}: {e}")
                    input_df[col] = 0
        
        print("🔢 DataFrame after encoding:\n", input_df)
        
        # Scale numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        print("⚖️ DataFrame after scaling:\n", input_df)
        
        # Make prediction
        prediction = int(model.predict(input_df)[0])
        probability = model.predict_proba(input_df)[0]
        
        churn_prob = float(probability[1]) * 100
        retention_prob = float(probability[0]) * 100
        
        print(f"✅ Prediction: {prediction}, Churn: {churn_prob:.2f}%")
        
        result = {
            'prediction': 'Will Churn' if prediction == 1 else 'Will Not Churn',
            'churn_probability': round(churn_prob, 2),
            'retention_probability': round(retention_prob, 2)
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("❌ ERROR:", error_trace)
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed. Please check all fields.'
        }), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)