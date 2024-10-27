import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from scipy import stats
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose
import shap
import warnings
warnings.filterwarnings("ignore")
from streamlit_option_menu import option_menu

# Separate file structure for modularity and clarity
from app_sections.summary_statistics import summary_statistics
from app_sections.advanced_statistics import advanced_statistics
from app_sections.temporal_analysis import temporal_analysis
from app_sections.exploratory_analysis import exploratory_analysis
from app_sections.outlier_analysis import outlier_analysis
from app_sections.user_guidance import user_guidance
from app_sections.regression_analysis import regression_analysis
from app_sections.feature_correlation import feature_correlation
from app_sections.feature_importance import feature_importance

st.set_page_config(page_title='Exploratory Data Analysis App', layout='wide')

# App Title
st.title('EDA Application')

# Sidebar for language selection
st.sidebar.title('Exploratory Data Analysis Application')
st.sidebar.markdown('## Select your preferred language')
language = st.sidebar.radio('Select language', ('English', 'Spanish'))

# Sample data option
def load_sample_data():
    dates = pd.date_range(start="2023-01-01", periods=365, freq='D')
    data = {
        'Date': dates,
        'PM2.5': np.random.uniform(20, 100, size=len(dates)),
        'PM10': np.random.uniform(30, 150, size=len(dates)),
        'NO2': np.random.uniform(10, 80, size=len(dates)),
        'CO': np.random.uniform(0.5, 2.5, size=len(dates)),
        'Temperature': np.random.uniform(10, 35, size=len(dates)),
        'Humidity': np.random.uniform(30, 90, size=len(dates))
    }
    return pd.DataFrame(data)

# File upload or sample data selection
data_source = st.sidebar.selectbox('Select Data Source', ('Upload CSV', 'Use Sample Data'))

data = None
if data_source == 'Upload CSV':
    data = st.file_uploader('Upload a CSV file', type='csv')
    if data is not None:
        df = pd.read_csv(data)
else:
    st.markdown('Using Sample Air Pollution Dataset')
    df = load_sample_data()
    st.write('Sample Air Pollution Data')
    st.write(df)

if data is not None or data_source == 'Use Sample Data':
    st.write('First 5 rows of the data')
    st.write(df.head())

    # Interface with Tabs
    if language == 'English':
        with st.sidebar:
            selected_tab = option_menu("Main Menu", ["Summary Statistics", "Advanced Statistics", "Temporal Analysis", "Exploratory Analysis", "Outlier Analysis", "Feature Correlation", "Regression Analysis", "Feature Importance", "User Guidance"], 
                                      icons=['bar-chart', 'clipboard-data', 'clock', 'binoculars', 'exclamation', 'link', 'line-chart', 'star', 'info-circle'], 
                                      menu_icon="cast", default_index=0)
        
        if selected_tab == "Summary Statistics":
            summary_statistics(df)
        elif selected_tab == "Advanced Statistics":
            advanced_statistics(df)
        elif selected_tab == "Temporal Analysis":
            temporal_analysis(df)
        elif selected_tab == "Exploratory Analysis":
            exploratory_analysis(df)
        elif selected_tab == "Outlier Analysis":
            outlier_analysis(df)
        elif selected_tab == "Feature Correlation":
            feature_correlation(df)
        elif selected_tab == "Regression Analysis":
            regression_analysis(df)
        elif selected_tab == "Feature Importance":
            feature_importance(df)
        elif selected_tab == "User Guidance":
            user_guidance()

    else:
        with st.sidebar:
            selected_tab = option_menu("Menú Principal", ["Estadísticas Descriptivas", "Estadísticas Avanzadas", "Análisis Temporal", "Análisis Exploratorio", "Análisis de Valores Atípicos", "Correlación de Características", "Análisis de Regresión", "Importancia de Características", "Guía del Usuario"], 
                                      icons=['bar-chart', 'clipboard-data', 'clock', 'binoculars', 'exclamation', 'link', 'line-chart', 'star', 'info-circle'], 
                                      menu_icon="cast", default_index=0)
        
        if selected_tab == "Estadísticas Descriptivas":
            summary_statistics(df, language='Spanish')
        elif selected_tab == "Estadísticas Avanzadas":
            advanced_statistics(df, language='Spanish')
        elif selected_tab == "Análisis Temporal":
            temporal_analysis(df, language='Spanish')
        elif selected_tab == "Análisis Exploratorio":
            exploratory_analysis(df, language='Spanish')
        elif selected_tab == "Análisis de Valores Atípicos":
            outlier_analysis(df, language='Spanish')
        elif selected_tab == "Correlación de Características":
            feature_correlation(df, language='Spanish')
        elif selected_tab == "Análisis de Regresión":
            regression_analysis(df, language='Spanish')
        elif selected_tab == "Importancia de Características":
            feature_importance(df, language='Spanish')
        elif selected_tab == "Guía del Usuario":
            user_guidance(language='Spanish')

# Files to be created:
#app_sections/summary_statistics.py
#app_sections/advanced_statistics.py
#app_sections/temporal_analysis.py
#app_sections/exploratory_analysis.py
#app_sections/outlier_analysis.py
#app_sections/feature_correlation.py
#app_sections/regression_analysis.py
#app_sections/feature_importance.py
#app_sections/user_guidance.py
