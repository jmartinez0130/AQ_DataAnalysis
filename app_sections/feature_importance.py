import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px


def feature_importance(df, language='English'):
    """
    Perform feature importance analysis using a RandomForest model.

    Parameters:
    df (DataFrame): The dataframe for which to perform feature importance analysis.
    language (str): The language to display ('English' or 'Spanish').
    """
    if language == 'English':
        st.markdown('## Feature Importance Analysis')

        # Select features and target variable
        feature_columns = st.multiselect('Select feature columns', df.columns)
        target_column = st.selectbox('Select target column', df.columns)

        if len(feature_columns) > 0 and target_column:
            X = df[feature_columns].dropna()
            y = df[target_column].loc[X.index]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Random Forest Regressor for Feature Importance
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Get feature importances
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importance})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            st.write(feature_importance_df)

            # Plotting Feature Importance
            fig = px.bar(feature_importance_df, x='Feature', y='Importance', title='Feature Importance')
            st.plotly_chart(fig)

        else:
            st.write("Please select feature columns and target column for feature importance analysis.")

    elif language == 'Spanish':
        st.markdown('## Análisis de Importancia de Características')

        # Seleccionar características y variable objetivo
        feature_columns = st.multiselect('Seleccione columnas de características', df.columns)
        target_column = st.selectbox('Seleccione la columna objetivo', df.columns)

        if len(feature_columns) > 0 and target_column:
            X = df[feature_columns].dropna()
            y = df[target_column].loc[X.index]

            # Dividir los datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Random Forest Regressor para la Importancia de Características
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Obtener la importancia de las características
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Característica': feature_columns, 'Importancia': importance})
            feature_importance_df = feature_importance_df.sort_values(by='Importancia', ascending=False)
            st.write(feature_importance_df)

            # Gráfica de Importancia de Características
            fig = px.bar(feature_importance_df, x='Característica', y='Importancia', title='Importancia de Características')
            st.plotly_chart(fig)

        else:
            st.write("Por favor seleccione columnas de características y columna objetivo para el análisis de importancia de características.")
