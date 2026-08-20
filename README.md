# ❤️ Heart Disease Risk Predictor

An AI-powered web application that predicts cardiovascular disease risk using machine learning, built with Python, Streamlit, and Scikit-learn.

---

## 🚀 Project Status

**Currently in active development** — building day by day with a focus on understanding every part of the pipeline, not just copy-pasting code.

- ✅ Project structure & environment setup
- ✅ Dependencies & configuration
- ✅ Data preprocessing pipeline
- ✅ Model training (Random Forest)
- 🔄 Streamlit web interface (in progress)
- 🔄 Deployment (upcoming)

---

## 📊 Dataset

- **Source:** [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Records:** 1,025 patients
- **Features:** 13 clinical attributes
- **Target:** Presence of heart disease (0 = No, 1 = Yes)

### Clinical Features Used

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia type |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest Classifier)
- **Visualization:** Plotly
- **Model Persistence:** Joblib
- **Version Control:** Git & GitHub

---

## 📁 Project Structure

```
heart-disease-predictor/
├── app/
│   ├── config.py              # App configuration & constants
│   ├── main.py                 # Streamlit entry point
│   ├── models/
│   │   ├── preprocessing.py    # Data loading, splitting, scaling
│   │   └── train_model.py      # Model training pipeline
│   └── pages/                  # Streamlit multi-page app
├── data/
│   └── raw/
│       └── heart_disease.csv   # Source dataset
├── models/
│   ├── trained_model.pkl       # Saved Random Forest model
│   └── scaler.pkl              # Saved StandardScaler
├── .streamlit/
│   └── config.toml             # Streamlit theme/server config
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/saad-abbasi01/heart-disease-predictor.git
   cd heart-disease-predictor
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Model Training

The preprocessing and model training pipeline can be run directly:

```bash
python app/models/preprocessing.py
python app/models/train_model.py
```

This will:
1. Load and validate the dataset
2. Split into training (70%) and testing (30%) sets
3. Scale features using `StandardScaler`
4. Train a `RandomForestClassifier`
5. Evaluate performance (Accuracy, Precision, Recall, F1-score)
6. Save the trained model and scaler to `models/`

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~99% |
| Precision | ~100% |
| Recall | ~98% |
| F1-Score | ~99% |

*(Performance metrics will be reviewed and refined as the project progresses.)*

---

## ⚠️ Disclaimer

This project is built for **educational purposes only**. It is **not** a medical diagnostic tool and should never be used as a substitute for professional medical advice.

---

## 👨‍💻 Author

**Saad Abbasi**
BSCS Student | Data Science & ML Enthusiast
[GitHub](https://github.com/saad-abbasi01)

---

## 📝 License

This project is open source and available under the MIT License.
