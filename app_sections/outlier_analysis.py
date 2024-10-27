import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def outlier_analysis(df, language='English'):
    """
    Perform outlier analysis using Isolation Forest and visualize the detected outliers.

    Parameters:
    df (DataFrame): The dataframe for which to perform outlier analysis.
    language (str): The language to display ('English' or 'Spanish').
    """
    if language == 'English':
        st.markdown('## Outlier Analysis')

        # Select columns for outlier detection
        outlier_columns = st.multiselect('Select columns for outlier analysis', df.columns)
        if st.button('Detect Outliers'):
            if len(outlier_columns) > 0:
                outlier_data = df[outlier_columns].dropna()
                scaler = StandardScaler()
                outlier_data_scaled = scaler.fit_transform(outlier_data)

                # Apply Isolation Forest
                isolation_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
                outliers = isolation_forest.fit_predict(outlier_data_scaled)
                df['Outliers'] = np.where(outliers == -1, 'Outlier', 'Normal')

                # Display the results
                st.write(df)
                fig = px.scatter_matrix(df, dimensions=outlier_columns, color='Outliers', title='Outlier Analysis Results')
                st.plotly_chart(fig)
            else:
                st.write("Please select at least one column for outlier analysis.")

    elif language == 'Spanish':
        st.markdown('## Análisis de Valores Atípicos')

        # Seleccione columnas para la detección de valores atípicos
        outlier_columns = st.multiselect('Seleccione columnas para el análisis de valores atípicos', df.columns)
        if st.button('Detectar Valores Atípicos'):
            if len(outlier_columns) > 0:
                outlier_data = df[outlier_columns].dropna()
                scaler = StandardScaler()
                outlier_data_scaled = scaler.fit_transform(outlier_data)

                # Aplicar Isolation Forest
                isolation_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
                outliers = isolation_forest.fit_predict(outlier_data_scaled)
                df['Valores Atípicos'] = np.where(outliers == -1, 'Valor Atípico', 'Normal')

                # Mostrar los resultados
                st.write(df)
                fig = px.scatter_matrix(df, dimensions=outlier_columns, color='Valores Atípicos', title='Resultados del Análisis de Valores Atípicos')
                st.plotly_chart(fig)
            else:
                st.write("Por favor seleccione al menos una columna para el análisis de valores atípicos.")
