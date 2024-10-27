import pandas as pd
import streamlit as st

def summary_statistics(df, language='English'):
    """
    Display summary statistics including percentage of missing data.

    Parameters:
    df (DataFrame): The dataframe for which to compute summary statistics.
    language (str): The language to display ('English' or 'Spanish').
    """
    if language == 'English':
        st.markdown('## Summary Statistics with Percentage of Missing Data')
        summary_stats = df.describe(include='all').T
        summary_stats['Percentage Missing'] = (df.isna().sum() / df.shape[0]) * 100
        summary_stats['Percentage Missing'] = summary_stats['Percentage Missing'].round(2)
        st.write(summary_stats)
    
    elif language == 'Spanish':
        st.markdown('## Estadísticas Descriptivas con Porcentaje de Datos Faltantes')
        summary_stats = df.describe(include='all').T
        summary_stats['Porcentaje Faltante'] = (df.isna().sum() / df.shape[0]) * 100
        summary_stats['Porcentaje Faltante'] = summary_stats['Porcentaje Faltante'].round(2)
        st.write(summary_stats)
