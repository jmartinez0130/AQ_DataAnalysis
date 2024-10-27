import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


def exploratory_analysis(df, language='English'):
    """
    Perform exploratory analysis including visualizations to understand data distributions and relationships.

    Parameters:
    df (DataFrame): The dataframe for which to perform exploratory analysis.
    language (str): The language to display ('English' or 'Spanish').
    """
    if language == 'English':
        st.markdown('## Exploratory Data Analysis')

        # Box Plots
        st.markdown('### Box Plots')
        box_columns = st.multiselect('Select columns to create box plots', df.columns)
        if st.button('Generate Box Plots'):
            for col in box_columns:
                fig, ax = plt.subplots()
                sns.boxplot(data=df, y=col, ax=ax)
                st.pyplot(fig)

        # Pair Plots
        st.markdown('### Pair Plots')
        pair_columns = st.multiselect('Select columns for pair plot', df.columns)
        if st.button('Generate Pair Plot'):
            if len(pair_columns) > 1:
                fig = sns.pairplot(df[pair_columns])
                st.pyplot(fig)
            else:
                st.write("Please select at least two columns for pair plot.")

        # Scatter Plot
        st.markdown('### Scatter Plot')
        x_axis = st.selectbox('Select the column for X axis', df.columns)
        y_axis = st.selectbox('Select the column for Y axis', df.columns)
        if st.button('Generate Scatter Plot'):
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot of {y_axis} vs {x_axis}')
            st.plotly_chart(fig)

    elif language == 'Spanish':
        st.markdown('## Análisis Exploratorio de Datos')

        # Diagramas de Caja
        st.markdown('### Diagramas de Caja')
        box_columns = st.multiselect('Seleccione columnas para crear diagramas de caja', df.columns)
        if st.button('Generar Diagramas de Caja'):
            for col in box_columns:
                fig, ax = plt.subplots()
                sns.boxplot(data=df, y=col, ax=ax)
                st.pyplot(fig)

        # Diagramas de Pares
        st.markdown('### Diagramas de Pares')
        pair_columns = st.multiselect('Seleccione columnas para el diagrama de pares', df.columns)
        if st.button('Generar Diagrama de Pares'):
            if len(pair_columns) > 1:
                fig = sns.pairplot(df[pair_columns])
                st.pyplot(fig)
            else:
                st.write("Por favor seleccione al menos dos columnas para el diagrama de pares.")

        # Diagrama de Dispersión
        st.markdown('### Diagrama de Dispersión')
        x_axis = st.selectbox('Seleccione la columna para el eje X', df.columns)
        y_axis = st.selectbox('Seleccione la columna para el eje Y', df.columns)
        if st.button('Generar Diagrama de Dispersión'):
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Diagrama de Dispersión de {y_axis} vs {x_axis}')
            st.plotly_chart(fig)
