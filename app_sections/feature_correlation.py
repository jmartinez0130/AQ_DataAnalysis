import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


def feature_correlation(df, language='English'):
    """
    Perform feature correlation analysis and visualize correlation heatmaps.

    Parameters:
    df (DataFrame): The dataframe for which to perform feature correlation analysis.
    language (str): The language to display ('English' or 'Spanish').
    """
    if language == 'English':
        st.markdown('## Feature Correlation Analysis')

        # Select columns for correlation analysis
        correlation_columns = st.multiselect('Select columns for correlation analysis', df.columns)
        if len(correlation_columns) > 1:
            # Compute correlation matrix
            correlation_matrix = df[correlation_columns].corr()
            st.write(correlation_matrix)

            # Display correlation heatmap
            st.markdown('### Correlation Heatmap')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', ax=ax)
            st.pyplot(fig)

            # Display interactive heatmap
            st.markdown('### Interactive Correlation Heatmap')
            fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='YlGnBu', title='Feature Correlation Heatmap')
            st.plotly_chart(fig)
        else:
            st.write("Please select at least two columns for correlation analysis.")

    elif language == 'Spanish':
        st.markdown('## Análisis de Correlación de Características')

        # Seleccione columnas para el análisis de correlación
        correlation_columns = st.multiselect('Seleccione columnas para el análisis de correlación', df.columns)
        if len(correlation_columns) > 1:
            # Calcular la matriz de correlación
            correlation_matrix = df[correlation_columns].corr()
            st.write(correlation_matrix)

            # Mostrar el mapa de calor de la correlación
            st.markdown('### Mapa de Calor de Correlación')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', ax=ax)
            st.pyplot(fig)

            # Mostrar mapa de calor interactivo
            st.markdown('### Mapa de Calor de Correlación Interactivo')
            fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='YlGnBu', title='Mapa de Calor de Correlación de Características')
            st.plotly_chart(fig)
        else:
            st.write("Por favor seleccione al menos dos columnas para el análisis de correlación.")
