import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def temporal_analysis(df, language='English'):
    """
    Perform temporal analysis, including seasonal decomposition.

    Parameters:
    df (DataFrame): The dataframe for which to perform temporal analysis.
    language (str): The language to display ('English' or 'Spanish').
    """
    if language == 'English':
        st.markdown('## Seasonal and Temporal Analysis')
        time_column = st.selectbox('Select time column for seasonal decomposition', df.columns)
        value_column = st.selectbox('Select value column for seasonal decomposition', df.columns)
        if st.button('Perform Seasonal Decomposition'):
            df[time_column] = pd.to_datetime(df[time_column])
            df.set_index(time_column, inplace=True)
            decomposition = seasonal_decompose(df[value_column], model='additive', period=12)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observed')
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            st.pyplot(fig)

    elif language == 'Spanish':
        st.markdown('## Análisis Estacional y Temporal')
        time_column = st.selectbox('Seleccione la columna de tiempo para descomposición estacional', df.columns)
        value_column = st.selectbox('Seleccione la columna de valores para descomposición estacional', df.columns)
        if st.button('Realizar Descomposición Estacional'):
            df[time_column] = pd.to_datetime(df[time_column])
            df.set_index(time_column, inplace=True)
            decomposition = seasonal_decompose(df[value_column], model='additive', period=12)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observado')
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Tendencia')
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Estacional')
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            st.pyplot(fig)
