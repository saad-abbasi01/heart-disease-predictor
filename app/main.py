import sys
import os
sys.path.append(os.getcwd())
import streamlit as st


from app.models.predictor import HeartDiseasePredictor

#set page configure
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)
st.title("❤️Heart Disease Predictor")
st.markdown("Enter patient clinal data below to assess heart disease risk.")
st.markdown("---")
# for only load at once
@st.cache_resource
def load_data():
    return HeartDiseasePredictor(
        "app/models/train_model.pkl",
        "models/scaler.pkl"
    )
    
predictor=load_data()

with st.form("patient_data"):
    col1,col2=st.columns(2)

    #now columns data in the form of form.
    with col1:
        
        age=st.slider("Age",18,100,45)
        sex=st.radio("SEX",["Male","Female"],horizontal=True)
        sex_val=1 if sex=="Male" else 0
        cp=st.selectbox("Chest Pain(0,3)",[0,1,2,3])
        trestbps=st.slider("Resting Blodd Pressure (mmHg)",80,200,120)
        chol=st.slider("Cholesterol level(mg/dl)",100,400,200)
        fbs=st.checkbox("Fasting Blood Sugar > 120(mg/dl)")
        fbs_val=1 if fbs else 0
    with col2:
        restecg=st.selectbox("Resting ECG(0,2)",[0,1,2])
        thalach=st.slider("Max Heart Rate",60,202,150)
        exang=st.checkbox("Exercise INduce Angina.")
        exang_val=1 if exang else 0
        oldpeak=st.slider("ST Depression",0.0,6.2,1.0,step=0.1)
        slope=st.selectbox("ST Slope(0,2)",[0,1,2])
        ca=st.slider("Major Vessel(0,4)",0,4,0)
        thal =st.selectbox("Thalassemia(0,3)",[0,1,2,3])
        
    submitted=st.form_submit_button("Predict Risk",use_container_width=True)
        
  # submitted       
if submitted:
    patient_data={
    'age': age, 'sex': sex_val, 'cp': cp, 'trestbps': trestbps,
    'chol': chol, 'fbs': fbs_val, 'restecg': restecg,
    'thalach': thalach, 'exang': exang_val, 'oldpeak': oldpeak,
    'slope': slope, 'ca': ca, 'thal': thal
    }

    risk_pct, risk_level = predictor.prepare(patient_data)

    st.markdown("--❤️--")
    st.subheader("📊 Prediction Result")
    st.metric("Risk Score", f"{risk_pct}%")
    st.markdown(f"### {risk_level}")
    