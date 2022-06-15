import streamlit as st
import pandas as pd
import DataAnalysis.app1
import ModelBuilding.app2

st.sidebar.markdown("# Browse a file:")
uploaded_file = st.sidebar.file_uploader(" ")
# DataAnalysis.app1.show(1)
container1= st.empty()
container1.header( "Data Science Platform\n" )
st.image("DS-image.jpg")
container2= st.empty()
container2.subheader( "Upload a dataset, select a task in the left navigation bar and get things done!" )
codecs=['latin_1','utf_8','ISO-8859-1']
global df
if uploaded_file is not None:
    file_type = uploaded_file.name.split( '.' )[-1]

    if file_type=='csv':
        for x in range( len( codecs ) ):
            try:
                df = pd.read_csv(uploaded_file,encoding = (codecs[x]),sep='[:,|;]',error_bad_lines=False)
            except:
                pass
    elif file_type=='xlsx':
        for x in range( len( codecs ) ):
            try:
                # global df
                df = pd.read_excel( uploaded_file)
                # df = pd.read_csv(uploaded_file,encoding = (codecs[x]))
            except:
                pass
    else:
        st.text("Please upload a csv or an excel file")


    st.sidebar.markdown("# Select a task:")
    choice=st.sidebar.radio(" ",['Data Analysis','Model Building'])
    container1.empty()
    container2.empty()
    if choice=='Data Analysis':
        DataAnalysis.app1.app_data_analysis(df)
    elif choice=='Model Building':
        ModelBuilding.app2.app_model_building(df)
