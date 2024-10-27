import streamlit as st


def user_guidance(language='English'):
    """
    Provide user guidance for using the EDA application.

    Parameters:
    language (str): The language to display ('English' or 'Spanish').
    """
    if language == 'English':
        st.markdown('## User Guidance and Tooltips')
        st.markdown('### How to Use This Application')
        st.write("This application allows you to explore your dataset using various data analysis techniques. Below are descriptions of some of the key functionalities:")
        st.write("- **Summary Statistics**: Provides descriptive statistics for each column, including missing data percentage.")
        st.write("- **Advanced Statistical Analysis**: Visualize data distributions and check for normality.")
        st.write("- **Seasonal and Temporal Analysis**: Decompose time series data into trend, seasonal, and residual components.")
        st.write("- **Exploratory Analysis**: Generate box plots, pair plots, and scatter plots to explore data distributions and relationships.")
        st.write("- **Outlier Analysis**: Detect anomalous data points using Isolation Forest and visualize the results.")
        st.write("- **Feature Correlation**: Analyze the correlation between features and visualize it using heatmaps.")
        st.write("- **Regression Analysis**: Perform regression analysis using Linear, Ridge, and Lasso models to predict target variables.")
        st.write("- **Feature Importance**: Identify which features have the most influence on a selected target variable using a Random Forest model.")
        st.write("- **Tooltips**: Hover over different elements for more information on what each feature does.")

    elif language == 'Spanish':
        st.markdown('## Guía del Usuario y Tooltips')
        st.markdown('### Cómo Usar Esta Aplicación')
        st.write("Esta aplicación le permite explorar su conjunto de datos utilizando diversas técnicas de análisis de datos. A continuación se presentan descripciones de algunas de las funcionalidades clave:")
        st.write("- **Estadísticas Descriptivas**: Proporciona estadísticas descriptivas para cada columna, incluido el porcentaje de datos faltantes.")
        st.write("- **Análisis Estadístico Avanzado**: Visualice distribuciones de datos y verifique la normalidad.")
        st.write("- **Análisis Estacional y Temporal**: Descomponga datos de series temporales en componentes de tendencia, estacionalidad y residuales.")
        st.write("- **Análisis Exploratorio**: Genere diagramas de caja, diagramas de pares y diagramas de dispersión para explorar las distribuciones y relaciones de los datos.")
        st.write("- **Análisis de Valores Atípicos**: Detecte puntos de datos anómalos utilizando Isolation Forest y visualice los resultados.")
        st.write("- **Correlación de Características**: Analice la correlación entre características y visualícela utilizando mapas de calor.")
        st.write("- **Análisis de Regresión**: Realice análisis de regresión utilizando modelos Lineales, Ridge y Lasso para predecir variables objetivo.")
        st.write("- **Importancia de Características**: Identifique qué características tienen mayor influencia en una variable objetivo seleccionada utilizando un modelo de Bosque Aleatorio.")
        st.write("- **Tooltips**: Pase el cursor sobre diferentes elementos para obtener más información sobre lo que hace cada característica.")