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
