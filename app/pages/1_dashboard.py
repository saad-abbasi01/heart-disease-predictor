import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.getcwd())
st.set_page_config(page_title='Dashboard',page_icon='📊',layout="centered")
st.title("Dataset & Model Dashboard ")
st.markdown("---")

@st.cache_resource
def load_data():
   return pd.read_csv("data/raw/heart_disease.csv")
    
df=load_data()

col1,col2,col3=st.columns(3)
col1.metric("Total patients:",f"{len(df)}")
col2.metric("Average age: ",f"{df['age'].mean():.1f}")
col3.metric("Disease cases:",int(df['target'].sum()))
st.markdown("---")

tab1,tab2=st.tabs(['Age Distribution','Risk Dustribution'])

with tab1:
    fig=px.histogram(df,x='age',nbins=20,title='Patient Age Distribution')
    fig.update_traces(marker_color='#FF6B6B')
    st.plotly_chart(fig,use_container_width=True)
    
with tab2:
    risk_counts=df['target'].value_counts().reset_index()
    risk_counts.columns=['Disease status','Count']
    risk_counts['Disease status']=risk_counts['Disease status'].map({0:' No disease ',1: 'Disease has'})
    fig=px.pie(risk_counts,values='Count',names='Disease status',color_discrete_sequence=['#8BC34A', '#F44336'])   
    st.plotly_chart(fig,use_container_width=True)
    
st.markdown('---')

st.subheader("Model performance")

col1,col2,col3,col4=st.columns(4)
col1.metric("Accuracy: ","99%")
col2.metric("Precision:","100%")
col3.metric("Recall:","98%")
col4.metric("F1_score:","99%")

st.info("Note: These metrics were affected by duplicate records in the source dataset — see project notes for details.")


from sklearn.metrics import confusion_matrix,roc_curve,auc
import joblib
import plotly.figure_factory as ff

# now we have to load model and data
def load_model_and_data():
    model=joblib.load("app/models/train_model.pkl")
    scaler=joblib.load("models/scaler.pkl")
    return model ,scaler
model,scaler=load_model_and_data()
feature_cols=["age","sex","cp","trestbps","chol","fbs",'restecg',"thalach","exang","oldpeak","slope","ca","thal"]

X=df[feature_cols]
Y=df['target']

#trained the model using pre train_model
X_scaled=scaler.transform(X.values)

Y_pred=model.predict(X_scaled)
Y_proba=model.predict_proba(X_scaled)[:,1]
tab1,tab2=st.tabs(["Confusion_matrix","ROC_curve"])

with tab1:
    cm=confusion_matrix(Y,Y_pred)
    labels=["Disease","No disease"]
    
    
    fig=ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True  
     )
    fig.update_layout(
        title="Confusion_matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"

    )
    st.plotly_chart(fig,user_container_width=True)
    tn,fp,fn,tp=cm.ravel()
    col1,col2,col3,col4=st.columns(4)
    
    col1.metric("True negative",tn)
    col2.metric("False positive",fp)
    col3.metric("False negative",fn)
    col4.metric("True postive",tp)
    
with tab2:
    fpr,tpr ,threshold=roc_curve(Y,Y_proba)
    roc_auc=auc(fpr,tpr)
    
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,y=tpr,mode='lines',name=f'Roc curve the auc is:{roc_auc:.2f}',
        line=dict(color='#FF3B3B',width=3)
    ))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',
                             name="Random Guess",
                             line=dict(color='grey',dash='dash',width=2)
                             ))
    fig.update_layout(
            title="Roc_curve",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate"
     )
    st.plotly_chart(fig,user_container_width=True)