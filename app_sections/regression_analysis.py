import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px


def regression_analysis(df, language='English'):
    """
    Perform regression analysis using Linear, Ridge, and Lasso regression models.

    Parameters:
    df (DataFrame): The dataframe for which to perform regression analysis.
    language (str): The language to display ('English' or 'Spanish').
    """
    if language == 'English':
        st.markdown('## Regression Analysis')

        # Select features and target variable
        feature_columns = st.multiselect('Select feature columns', df.columns)
        target_column = st.selectbox('Select target column', df.columns)

        if len(feature_columns) > 0 and target_column:
            X = df[feature_columns].dropna()
            y = df[target_column].loc[X.index]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Linear Regression
            st.markdown('### Linear Regression')
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred_linear = linear_model.predict(X_test)
            st.write(f"R^2 Score: {r2_score(y_test, y_pred_linear):.2f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_linear):.2f}")

            # Ridge Regression
            st.markdown('### Ridge Regression')
            ridge_model = Ridge()
            ridge_model.fit(X_train, y_train)
            y_pred_ridge = ridge_model.predict(X_test)
            st.write(f"R^2 Score: {r2_score(y_test, y_pred_ridge):.2f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_ridge):.2f}")

            # Lasso Regression
            st.markdown('### Lasso Regression')
            lasso_model = Lasso()
            lasso_model.fit(X_train, y_train)
            y_pred_lasso = lasso_model.predict(X_test)
            st.write(f"R^2 Score: {r2_score(y_test, y_pred_lasso):.2f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lasso):.2f}")

            # Plotting Predictions vs Actual
            st.markdown('### Predictions vs Actual')
            comparison_df = pd.DataFrame({
                'Actual': y_test,
                'Linear Prediction': y_pred_linear,
                'Ridge Prediction': y_pred_ridge,
                'Lasso Prediction': y_pred_lasso
            }).reset_index(drop=True)
            fig = px.line(comparison_df, title='Regression Predictions vs Actual Values')
            st.plotly_chart(fig)

        else:
            st.write("Please select feature columns and target column for regression analysis.")

    elif language == 'Spanish':
        st.markdown('## Análisis de Regresión')

        # Seleccionar características y variable objetivo
        feature_columns = st.multiselect('Seleccione columnas de características', df.columns)
        target_column = st.selectbox('Seleccione la columna objetivo', df.columns)

        if len(feature_columns) > 0 and target_column:
            X = df[feature_columns].dropna()
            y = df[target_column].loc[X.index]

            # Dividir los datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Regresión Lineal
            st.markdown('### Regresión Lineal')
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred_linear = linear_model.predict(X_test)
            st.write(f"Coeficiente de determinación (R^2): {r2_score(y_test, y_pred_linear):.2f}")
            st.write(f"Error Cuadrático Medio: {mean_squared_error(y_test, y_pred_linear):.2f}")

            # Regresión Ridge
            st.markdown('### Regresión Ridge')
            ridge_model = Ridge()
            ridge_model.fit(X_train, y_train)
            y_pred_ridge = ridge_model.predict(X_test)
            st.write(f"Coeficiente de determinación (R^2): {r2_score(y_test, y_pred_ridge):.2f}")
            st.write(f"Error Cuadrático Medio: {mean_squared_error(y_test, y_pred_ridge):.2f}")

            # Regresión Lasso
            st.markdown('### Regresión Lasso')
            lasso_model = Lasso()
            lasso_model.fit(X_train, y_train)
            y_pred_lasso = lasso_model.predict(X_test)
            st.write(f"Coeficiente de determinación (R^2): {r2_score(y_test, y_pred_lasso):.2f}")
            st.write(f"Error Cuadrático Medio: {mean_squared_error(y_test, y_pred_lasso):.2f}")

            # Gráfica de Predicciones vs Valores Reales
            st.markdown('### Predicciones vs Valores Reales')
            comparison_df = pd.DataFrame({
                'Valor Real': y_test,
                'Predicción Lineal': y_pred_linear,
                'Predicción Ridge': y_pred_ridge,
                'Predicción Lasso': y_pred_lasso
            }).reset_index(drop=True)
            fig = px.line(comparison_df, title='Predicciones de Regresión vs Valores Reales')
            st.plotly_chart(fig)

        else:
            st.write("Por favor seleccione columnas de características y columna objetivo para el análisis de regresión.")
