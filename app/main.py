import sys
import os
sys.path.append(os.getcwd())
import streamlit as st


from app.models.predictor import HeartDiseasePredictor
import plotly.graph_objects as go
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
    col_a,col_b=st.columns([1,1])
    
    with col_a:
        fig=go.Figure(go.Indicator(
            mode="gauge + number",
            value=risk_pct,
            title={"text":"Risk_Score (%)"},
            gauge={
                
                'axis':{'range':[0,100]},
                'bar':{'color':"#333333"},
                'steps':[
                    {'range':[0,30],'color':'#8BC34A'},
                    {'range':[30,70],'color':'#FFC107'},
                    {'range':[70,100],'color':'#F44336'}
                    
                ],
                
            }
        ))
        fig.update_layout(height=280,margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with col_b:
        st.markdown(f"###{risk_level}###")
        st.metric("Risk_percentage",f"{risk_pct:.2f}%")
        
        importances=predictor.model.feature_importances_
        top_features=sorted(zip(predictor.feature_cols,importances),key=lambda x: x[1],reverse=True)[:5]
        
        st.markdown("**Top  5 Factor influences this model**")
        for name ,score in  top_features:
            st.write(f" - {name} : {score:.2%}")
            
        st.markdown("---")
        st.subheader("Why predict this risk ratio?")
        
        explanation=predictor.explain(patient_data)
        explanation_sorted=sorted(explanation,key=lambda x: abs(x[1]),reverse=True)[:5]
        
        for name ,value in explanation_sorted:
            direction="Increased" if value >0 else "Decreased"
            st.write(f"**{name}** and {direction} this patient risk by {abs(value)}:.3f")
        