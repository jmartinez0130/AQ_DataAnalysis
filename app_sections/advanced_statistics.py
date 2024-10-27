import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def advanced_statistics(df, language='English'):
    """
    Display advanced statistical analysis, including distribution plots and normality checks.

    Parameters:
    df (DataFrame): The dataframe for which to compute advanced statistics.
    language (str): The language to display ('English' or 'Spanish').
    """
    if language == 'English':
        st.markdown('## Advanced Statistical Analysis')

        # Distribution Plots
        st.markdown('### Distribution Plots')
        dist_columns = st.multiselect('Select columns for distribution plots', df.columns)
        if st.button('Generate Distribution Plots'):
            for col in dist_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)

        # Normality Check
        st.markdown('### Normality Check')
        normality_column = st.selectbox('Select column for normality check', df.columns)
        if st.button('Perform Normality Check'):
            fig, ax = plt.subplots()
            stats.probplot(df[normality_column].dropna(), dist="norm", plot=ax)
            st.pyplot(fig)
            p_value = stats.shapiro(df[normality_column].dropna())[1]
            st.write("Shapiro-Wilk Test p-value:", p_value)
            if p_value < 0.05:
                st.write("The data is not normally distributed.")
            else:
                st.write("The data is normally distributed.")

    elif language == 'Spanish':
        st.markdown('## Análisis Estadístico Avanzado')

        # Gráficos de Distribución
        st.markdown('### Gráficos de Distribución')
        dist_columns = st.multiselect('Seleccione columnas para los gráficos de distribución', df.columns)
        if st.button('Generar Gráficos de Distribución'):
            for col in dist_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)

        # Prueba de Normalidad
        st.markdown('### Prueba de Normalidad')
        normality_column = st.selectbox('Seleccione columna para la prueba de normalidad', df.columns)
        if st.button('Realizar Prueba de Normalidad'):
            fig, ax = plt.subplots()
            stats.probplot(df[normality_column].dropna(), dist="norm", plot=ax)
            st.pyplot(fig)
            p_value = stats.shapiro(df[normality_column].dropna())[1]
            st.write("Valor p de la prueba de Shapiro-Wilk:", p_value)
            if p_value < 0.05:
                st.write("Los datos no están distribuidos normalmente.")
            else:
                st.write("Los datos están distribuidos normalmente.")
