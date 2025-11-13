## End to End machine learning project

# 🎯 Customer Churn Prediction System

A complete end-to-end machine learning project that predicts customer churn for telecom companies using real-world data.

## 🌟 Live Demo
[🔗 Try the app here](your-app-url-will-go-here)

## 📊 Project Overview
This system predicts whether a customer will leave (churn) based on:
- Contract type and tenure
- Service usage patterns
- Billing information
- Demographics

**Dataset**: IBM Telco Customer Churn (7,043 customers, 19 features)

## 🎯 Model Performance
- **Best Model**: Random Forest / Gradient Boosting
- **Accuracy**: ~80%
- **ROC-AUC**: ~84%
- **F1-Score**: ~60%

## 🛠️ Tech Stack
- **ML/Data Science**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Flask
- **Deployment**: Render
- **Language**: Python 3.10

## 📁 Project Structure
```
├── app.py                    # Flask web application
├── train_model.py            # Model training pipeline
├── preprocessing.py          # Data preprocessing
├── eda.py                    # Exploratory data analysis
├── templates/
│   └── index.html           # Web interface
├── model.pkl          # Trained model
├── scaler.pkl               # Feature scaler
├── label_encoders.pkl       # Categorical encoders
└── requirements.txt         # Dependencies
```

## 🚀 Local Setup

### Prerequisites
- Python 3.9 or higher

### Installation
1. Clone the repository
```bash
git clone <your-repo-url>
cd churn-prediction-project
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app.py
```

5. Open browser at `http://localhost:5000`

## 📈 Training Your Own Model
```bash
# 1. Explore data
python eda.py

# 2. Preprocess data
python preprocessing.py

# 3. Train models
python train_model.py

# 4. Run app with new model
python app.py
```

## 🎓 Key Features
- ✅ Real-world dataset from IBM
- ✅ Complete ML pipeline (EDA → Training → Deployment)
- ✅ Interactive web interface
- ✅ Model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- ✅ Feature importance analysis
- ✅ Production-ready deployment


## 🤝 Contributing
Feel free to fork this project and submit pull requests!

## 📄 License
MIT License

## 👤 Author
**Your Name**
- GitHub: [@yourusername](https://github.com/hshk2003)
- LinkedIn: [Your LinkedIn](www.linkedin.com/in/shaikh-huzaifa-b8a143334)

## 🙏 Acknowledgments
- Dataset: IBM Telco Customer Churn
- Inspiration: Real-world business problem solving
```

---

#### Update `requirements.txt` for deployment:
Replace your `requirements.txt` with this production-ready version:
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
flask==2.3.3
joblib==1.3.2
gunicorn==21.2.0