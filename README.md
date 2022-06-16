# Data Science Platform

Link of the web application: 
https://data-science-platform.herokuapp.com/

Data Science Projects demands majority of time in Data Cleaning, Data Preprocessing, Exploratory Data Analysis (EDA) and gaining valuable insights. Only after perfoming these steps a base line model is built which can be further improved. By automating these tasks, we can significantly reduce the time required to analyze and build a base line model.   

This project "Data Science Platform" is created in Python using several Data Science libraries such as Pandas, Numpy, Plotly, Sci-kit Learn, etc. The Front-end is created using Streamlit library and is deployed on Heroku.
 ### Data analysis:
 Data analysis is the act of analysing, cleaning, plotting and translating it into information in order to identify usable information, and aid decision-making.

This platform displays various dataset information such as 
- Missing values
- Duplicates 
- Outliers using IQR and Percentile methods

An user can create interactive plots created with Plotly library just by selecting features. Some of the plots are
- Histogram plot
- Scatter plot
- Cumulative Distribution Function (CDF) plot
- Correlation heatmaps
 and many more!

### Model Building:
Building a Machine Learning model is a process that involves feeding the model with training data from which it learns and then predicts new data points.
 
 This platform handles the preprocessing steps such as 
 - Missing value treatment
    - Numerical features:
        
            1. Mean 
            2. Median
            3. Zero (0)
    - Categorical features:
        
            1. Mode
            2. Unknown 
 - Outliers treatment
        
            1. IQR
            2. Percentile
 - Encoding

            1. One Hot 
            2. Ordinal
 - Scaling

            1. Min Max Scaler
            2. Standard Scaler
            3. Robust Scaler

after perfoming the above mentioned steps, data will be splitted into training and testing datasets for model building

Some of the models included in this platform are
1. Linear and Logistic Regression
2. Decision Trees
3. Several Ensemble Models


