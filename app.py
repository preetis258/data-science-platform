import streamlit as st
import pandas as pd
from DataAnalysis import app1
from ModelBuilding import app2

st.sidebar.markdown("# Browse a file:")
uploaded_file = st.sidebar.file_uploader(" ")
container1= st.empty()
container1.header( "Data Science Platform\n" )
st.image("DS-image.jpg")
container2= st.empty()
container2.subheader( "Upload a dataset, select a task in the left navigation bar and get things done!" )
encodings=['latin_1','utf_8','ISO-8859-1']
global df
if uploaded_file is not None:
    file_type = uploaded_file.name.split( '.' )[-1]

    if file_type=='csv':
        for x in range( len( encodings ) ):
            try:
                df = pd.read_csv(uploaded_file,encoding = (encodings[x]),sep='[:,|;]',error_bad_lines=False)
            except:
                pass
    elif file_type=='xlsx':
            df = pd.read_excel( uploaded_file)
    else:
        st.text("Please upload a csv or an excel file")


    st.sidebar.markdown("# Select a task:")
    choice=st.sidebar.radio(" ",['Data Analysis','Model Building'])
    container1.empty()
    container2.empty()
    if choice=='Data Analysis':
        app1.app_data_analysis(df)
    elif choice=='Model Building':
        app2.app_model_building(df)
