# ======================================================================================================================
# TELCO CHURN PREDICTOR - STREAMLIT APP
# Machine Learning Final Exam - Master's in Data Science
# ======================================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from streamlit_extras.metric_cards import style_metric_cards
import streamlit_antd_components as sac

# Importar clases personalizadas desde utils
from utils import (
    TotalChargesCleaner,
    CustomerIDDropper,
    TotalChargesDropper,
    CategoricalRedundancyCleaner,
    CustomOneHotEncoder,
    CompletePreprocessingPipeline,
    DataFramePreparer
)

# ======================================================================================================================
# CONFIGURACIÓN DE PÁGINA
# ======================================================================================================================

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="assets/analytics.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================================================
# TÍTULO PRINCIPAL
# ======================================================================================================================

st.title(":material/monitoring: Telco Customer Churn Prediction")
st.markdown("---")
st.markdown("""
### Machine Learning Ensemble Models Dashboard
This application allows you to predict customer churn using three different ensemble models,
compare their performance, and explore the dataset.
""")

# st.markdown("---")
st.caption(":material/check_circle: App initialized successfully!")



# ======================================================================================================================
# FUNCIONES DE CARGA DE DATOS
# ======================================================================================================================

@st.cache_resource
def load_models():
    """Carga los 6 modelos entrenados"""
    with open('models/final_models_for_streamlit.pkl', 'rb') as f:
        models = pickle.load(f)
    return models


@st.cache_resource
def load_pipeline():
    """Carga el pipeline de preprocesamiento completo"""
    with open('models/complete_preprocessing_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


@st.cache_resource
def load_feature_importance():
    """Carga los datos de importancia de features"""
    with open('models/feature_importance_data.pkl', 'rb') as f:
        feature_importance = pickle.load(f)
    return feature_importance


@st.cache_data
def load_metrics():
    """Carga las métricas calculadas en test set"""
    with open('data/final_metrics_for_streamlit.pkl', 'rb') as f:
        metrics = pickle.load(f)
    return metrics


@st.cache_data
def load_feature_config():
    """Carga la configuración de features (all vs selected)"""
    with open('data/feature_config_for_streamlit.pkl', 'rb') as f:
        config = pickle.load(f)
    return config


@st.cache_data
def load_summary_table():
    """Carga la tabla resumen comparativa"""
    with open('data/final_summary_table.pkl', 'rb') as f:
        summary = pickle.load(f)
    return summary


@st.cache_data
def load_prepared_data():
    """Carga los datos preparados para EDA"""
    with open('data/telco_data_prepared.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

@st.cache_data
def load_raw_data():
    """Carga el dataset original sin preprocesar para EDA"""
    try:
        df_raw = pd.read_csv('data/telcoCustomer.csv')
        return df_raw
    except FileNotFoundError:
        st.warning("telcoCustomer.csv not found. Using preprocessed data for EDA.")
        return None

# ======================================================================================================================
# CARGAR TODOS LOS DATOS AL INICIO
# ======================================================================================================================

try:
    models_dict = load_models()
    preprocessing_pipeline = load_pipeline()
    feature_importance_data = load_feature_importance()
    metrics_dict = load_metrics()
    feature_config = load_feature_config()
    summary_table = load_summary_table()
    prepared_data = load_prepared_data()
    df_raw_for_eda = load_raw_data()

    # st.success(":material/check_circle: All models and data loaded successfully!")
    st.caption(":material/check_circle: Application ready • Models loaded • Data initialized")

    st.markdown("---")

except Exception as e:
    st.error(f" Error loading data: {str(e)}")
    st.stop()


# ======================================================================================================================
# SIDEBAR - SELECTORES
# ======================================================================================================================

st.sidebar.markdown("---")
st.sidebar.title(":material/tune: Model Selection")


# Selector de tipo de modelo
model_type = st.sidebar.selectbox(
    "Select Model Type:",
    options=["Logistic Regression", "Stacking Classifier", "Voting Classifier (Soft)"],
    index=0
)

# Selector de versión (All Features vs Selected Features)
model_version = st.sidebar.selectbox(
    "Select Feature Version:",
    options=["All Features (22)", "Selected Features (12) - Optimized"],
    index=0
)

# st.sidebar.markdown("---")

# Mapeo de selecciones a nombres de archivos
model_mapping = {
    ("Logistic Regression", "All Features (22)"): "lr_all_features",
    ("Logistic Regression", "Selected Features (12) - Optimized"): "lr_selected_optimized",
    ("Stacking Classifier", "All Features (22)"): "stacking_all_features",
    ("Stacking Classifier", "Selected Features (12) - Optimized"): "stacking_selected_optimized",
    ("Voting Classifier (Soft)", "All Features (22)"): "voting_all_features",
    ("Voting Classifier (Soft)", "Selected Features (12) - Optimized"): "voting_selected_optimized",
}

metrics_mapping = {
    ("Logistic Regression", "All Features (22)"): "lr_all",
    ("Logistic Regression", "Selected Features (12) - Optimized"): "lr_opt",
    ("Stacking Classifier", "All Features (22)"): "stacking_all",
    ("Stacking Classifier", "Selected Features (12) - Optimized"): "stacking_opt",
    ("Voting Classifier (Soft)", "All Features (22)"): "voting_all",
    ("Voting Classifier (Soft)", "Selected Features (12) - Optimized"): "voting_opt",
}

# Obtener modelo y métricas seleccionadas
selected_model_key = model_mapping[(model_type, model_version)]
selected_metrics_key = metrics_mapping[(model_type, model_version)]

selected_model = models_dict[selected_model_key]
selected_metrics = metrics_dict[selected_metrics_key]

# Mostrar información del modelo seleccionado en sidebar
st.sidebar.success(f":material/check_circle: Model loaded: **{model_type}**")
st.sidebar.info(f":material/label: Version: **{model_version}**")

# Mostrar métricas del modelo seleccionado
st.sidebar.markdown("---")
st.sidebar.markdown("# :material/planner_review: Model Performance")

with st.sidebar:

    st.metric(label="Accuracy", value=f"{selected_metrics['acc']:.4f}")
    st.metric(label="AUC Score", value=f"{selected_metrics['auc']:.4f}")
    st.metric(label="F1-Score", value=f"{selected_metrics['f1']:.4f}")

    # 2. Aplicamos el estilo de tarjeta con la librería
    style_metric_cards(
        background_color="#1E1E1E",  # Gris muy oscuro (casi negro) para el fondo de la tarjeta
        border_left_color="#2196F3", # Verde Neón para el borde izquierdo (el resalte)
        border_color="#2E2E2E",      # Borde sutil gris para el resto del recuadro
        box_shadow=True              # Sombra suave para dar profundidad
    )

st.sidebar.markdown("---")

# ======================================================================================================================
# TABS PRINCIPALES
# ======================================================================================================================

selected_tab = sac.tabs(
    [
        sac.TabsItem(label='Make Prediction', icon='stars'),
        sac.TabsItem(label='Model Dashboard', icon='bar-chart-line-fill'),
        sac.TabsItem(label='EDA', icon='pie-chart'),
    ],
    align='left',
    size='lg',
    variant='segmented',
    color='blue' # Puedes cambiar a 'green', 'indigo', 'red', etc.
)

# ============================================================================
# TAB 1: FORMULARIO DE INFERENCIA
# ============================================================================

if selected_tab == 'Make Prediction':
    with st.container(border=True): # Opcional: un borde alrededor del contenido
        st.header(":material/auto_awesome_mosaic: Customer Churn Prediction")
        st.markdown("Enter customer information to predict churn probability.")
        st.markdown("---")

        # Crear formulario en 3 columnas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(":material/person: Customer Info")

            # Campo calculado: Customer ID
            st.text_input("Customer ID", value="PRED-001", disabled=True,
                          help="Auto-generated ID for prediction")

            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

            st.subheader(":material/phone: Phone Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        with col2:
            st.subheader(":material/wifi: Internet Services")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            st.subheader(":material/credit_card: Account Info")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method",
                                          ["Electronic check", "Mailed check",
                                           "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0,
                                              value=70.0, step=0.1)

            # Campo calculado: Total Charges
            total_charges_calculated = monthly_charges * tenure
            st.number_input("Total Charges ($)", value=float(total_charges_calculated),
                            disabled=True, help="Auto-calculated: Monthly Charges × Tenure")

        st.markdown("---")

        # Botón de predicción
        if st.button(":material/bolt: Predict Churn", type="primary", width='stretch'):

            # Crear DataFrame con los datos ingresados
            input_data = pd.DataFrame({
                'customerID': ['PRED-001'],
                'gender': [gender],
                'SeniorCitizen': [senior_citizen],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [str(total_charges_calculated)]  # string (object)
            })

            try:
                # Preprocesar datos
                with st.spinner("Processing data..."):
                    input_prep = preprocessing_pipeline.transform(input_data)

                # Filtrar features según versión seleccionada
                if "Selected" in model_version:
                    input_prep = input_prep[feature_config['selected_features']]

                # Realizar predicción
                with st.spinner("Making prediction..."):
                    prediction = selected_model.predict(input_prep)[0]
                    prediction_proba = selected_model.predict_proba(input_prep)[0]

                # Mostrar resultados
                st.markdown("---")
                st.subheader(":material/center_focus_strong: Prediction Results")

                # Calcular probabilidades
                churn_prob = prediction_proba[1] if hasattr(prediction_proba, '__len__') else prediction_proba
                retention_prob = 1 - churn_prob

                # Crear 2 columnas para los resultados
                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    if prediction == "Yes":
                        st.error("### :material/warning: CUSTOMER WILL CHURN")
                    else:
                        st.success("### :material/bookmark_check: CUSTOMER WILL STAY")

                    st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")

                with result_col2:
                    if prediction == "Yes":
                        st.error("### :material/warning: CHURN RISK: HIGH")
                    else:
                        st.success("### :material/bookmark_check: CHURN RISK: LOW")

                    st.metric("Retention Probability", f"{retention_prob * 100:.2f}%")


                # Mostrar probabilidades como barra de progreso
                # st.markdown("---")
                # st.markdown("**Probability Distribution:**")
                # st.progress(float(churn_prob),
                #             text=f"Churn: {churn_prob * 100:.1f}% | Retention: {retention_prob * 100:.1f}%")

                # Información adicional
                with st.expander(":material/description: View Input Data"):
                    # Crear tabla de visualización formateada (TODO como strings)
                    display_dict = {
                        'Feature': [],
                        'Value': []
                    }

                    display_dict['Feature'].extend(['Gender', 'Senior Citizen', 'Partner', 'Dependents',
                                                    'Tenure (months)', 'Phone Service', 'Multiple Lines',
                                                    'Internet Service', 'Online Security', 'Online Backup',
                                                    'Device Protection', 'Tech Support', 'Streaming TV',
                                                    'Streaming Movies', 'Contract', 'Paperless Billing',
                                                    'Payment Method', 'Monthly Charges', 'Total Charges (calculated)'])

                    display_dict['Value'].extend([
                        str(gender),
                        'Yes' if senior_citizen == 1 else 'No',
                        str(partner),
                        str(dependents),
                        str(tenure),
                        str(phone_service),
                        str(multiple_lines),
                        str(internet_service),
                        str(online_security),
                        str(online_backup),
                        str(device_protection),
                        str(tech_support),
                        str(streaming_tv),
                        str(streaming_movies),
                        str(contract),
                        str(paperless_billing),
                        str(payment_method),
                        f"${monthly_charges:.2f}",
                        f"${total_charges_calculated:.2f}"
                    ])

                    display_df = pd.DataFrame(display_dict)
                    # Asegurar que ambas columnas sean string
                    display_df['Feature'] = display_df['Feature'].astype(str)
                    display_df['Value'] = display_df['Value'].astype(str)

                    st.dataframe(display_df, width='stretch', hide_index=True)



            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)



# ----------------------------------------------------------------------------------------------------------------------
# SECCIÓN 5: Dashboard - Comparación de Métricas y Confusion Matrix
# ----------------------------------------------------------------------------------------------------------------------

# ============================================================================
# TAB 2: DASHBOARD INTERACTIVO
# ============================================================================

elif selected_tab == 'Model Dashboard':
    with st.container(border=True):
        st.header(":material/dashboard: Model Performance Dashboard")
        st.markdown("Explore model metrics, confusion matrices, feature importance, and dataset insights.")
        st.markdown("---")

        # ========================================================================
        # SECCIÓN 1: COMPARACIÓN DE MÉTRICAS
        # ========================================================================

        st.subheader(":material/compare: Performance Metrics Comparison")
        st.markdown(f"**Selected Model:** {model_type}")

        # Obtener métricas de ambas versiones del modelo seleccionado
        if "Logistic" in model_type:
            metrics_all = metrics_dict['lr_all']
            metrics_opt = metrics_dict['lr_opt']
            model_name_short = "LR"
        elif "Stacking" in model_type:
            metrics_all = metrics_dict['stacking_all']
            metrics_opt = metrics_dict['stacking_opt']
            model_name_short = "Stacking"
        else:  # Voting
            metrics_all = metrics_dict['voting_all']
            metrics_opt = metrics_dict['voting_opt']
            model_name_short = "Voting"

        # Crear DataFrame para comparación
        comparison_df = pd.DataFrame({
            'Metric': ['Accuracy', 'AUC', 'F1-Score'],
            'All Features (22)': [
                metrics_all['acc'],
                metrics_all['auc'],
                metrics_all['f1']
            ],
            'Selected Features (12)': [
                metrics_opt['acc'],
                metrics_opt['auc'],
                metrics_opt['f1']
            ]
        })

        # Crear gráfico de barras comparativo
        fig_comparison = go.Figure()

        fig_comparison.add_trace(go.Bar(
            name='All Features (22)',
            x=comparison_df['Metric'],
            y=comparison_df['All Features (22)'],
            marker_color='#00CC96',
            text=[f"{val:.4f}" for val in comparison_df['All Features (22)']],
            textposition='auto',
            textfont=dict(size=18)
        ))

        fig_comparison.add_trace(go.Bar(
            name='Selected Features (12)',
            x=comparison_df['Metric'],
            y=comparison_df['Selected Features (12)'],
            marker_color='#3498DB',
            text=[f"{val:.4f}" for val in comparison_df['Selected Features (12)']],
            textposition='auto',
            textfont=dict(size=18)
        ))

        fig_comparison.update_layout(
            title=dict(
                text=f"{model_name_short}: All Features vs Selected Features",
                font=dict(size=20)  # ← Tamaño del título
            ),
            xaxis=dict(
                title=dict(text="Metric", font=dict(size=18)),  # ← Tamaño título eje X
                tickfont=dict(size=14)  # ← Tamaño labels eje X
            ),
            yaxis=dict(
                title=dict(text="Score", font=dict(size=18)),  # ← Tamaño título eje Y
                tickfont=dict(size=14),  # ← Tamaño labels eje Y
                range=[0, 1.05]
            ),
            barmode='group',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=14)  # ← Tamaño texto de leyenda
            )
        )

        st.plotly_chart(fig_comparison, width='stretch')

        # Mostrar tabla comparativa
        with st.expander(":material/description: View Detailed Metrics Table"):
            # Formatear tabla
            comparison_display = comparison_df.copy()
            comparison_display['All Features (22)'] = comparison_display['All Features (22)'].apply(
                lambda x: f"{x:.4f}")
            comparison_display['Selected Features (12)'] = comparison_display['Selected Features (12)'].apply(
                lambda x: f"{x:.4f}")
            st.dataframe(comparison_display, width='stretch', hide_index=True)

        st.markdown("---")

        # ========================================================================
        # SECCIÓN 2: CONFUSION MATRICES
        # ========================================================================

        st.subheader(":material/apps: Confusion Matrix")

        # Crear dos columnas para las matrices
        cm_col1, cm_col2 = st.columns(2)

        with cm_col1:
            st.markdown("**All Features (22)**")
            cm_all = metrics_all['cm']

            # Crear heatmap con plotly
            fig_cm_all = go.Figure(data=go.Heatmap(
                z=cm_all,
                x=['Predicted: No', 'Predicted: Yes'],
                y=['Actual: No', 'Actual: Yes'],
                text=cm_all,
                texttemplate='%{text}',
                textfont={"size": 28},
                colorscale='Blues',
                showscale=False
            ))

            fig_cm_all.update_layout(
                title=dict(
                    text=f"{model_name_short} - All Features",
                    font=dict(size=20)  # ← Tamaño del título
                ),
                xaxis=dict(
                    title=dict(text="Predicted", font=dict(size=18)),  # ← Título eje X
                    tickfont=dict(size=14)  # ← Labels eje X
                ),
                yaxis=dict(
                    title=dict(text="Actual", font=dict(size=18)),  # ← Título eje Y
                    tickfont=dict(size=14),  # ← Labels eje Y
                    autorange='reversed'
                ),
                height=441, #350
            )

            st.plotly_chart(fig_cm_all, width='stretch')

        with cm_col2:
            st.markdown("**Selected Features (12)**")
            cm_opt = metrics_opt['cm']

            # Crear heatmap con plotly
            fig_cm_opt = go.Figure(data=go.Heatmap(
                z=cm_opt,
                x=['Predicted: No', 'Predicted: Yes'],
                y=['Actual: No', 'Actual: Yes'],
                text=cm_opt,
                texttemplate='%{text}',
                textfont={"size": 28},
                colorscale='Reds',
                showscale=False
            ))

            fig_cm_opt.update_layout(
                title=dict(
                    text=f"{model_name_short} - Selected Features",
                    font=dict(size=20)  # ← Tamaño del título
                ),
                xaxis=dict(
                    title=dict(text="Predicted", font=dict(size=18)),  # ← Título eje X
                    tickfont=dict(size=14)  # ← Labels eje X
                ),

                yaxis=dict(
                    title=dict(text="Actual", font=dict(size=18)),  # ← Título eje Y
                    tickfont=dict(size=14),  # ← Labels eje Y
                    autorange='reversed'
                ),
                height=441,
            )

            st.plotly_chart(fig_cm_opt, width='stretch')

        # Expander único DESPUÉS de ambas columnas
        with st.expander(":material/info: View Detailed Matrix Metrics"):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**All Features (22)**")
                tn, fp, fn, tp = cm_all.ravel()
                metric_col_a, metric_col_b = st.columns(2)
                with metric_col_a:
                    st.metric("True Negatives (TN)", tn)
                    st.metric("False Positives (FP)", fp)
                with metric_col_b:
                    st.metric("True Positives (TP)", tp)
                    st.metric("False Negatives (FN)", fn)

            with detail_col2:
                st.markdown("**Selected Features (12)**")
                tn, fp, fn, tp = cm_opt.ravel()
                metric_col_a, metric_col_b = st.columns(2)
                with metric_col_a:
                    st.metric("True Negatives (TN)", tn)
                    st.metric("False Positives (FP)", fp)
                with metric_col_b:
                    st.metric("True Positives (TP)", tp)
                    st.metric("False Negatives (FN)", fn)

        st.markdown("---")

# ----------------------------------------------------------------------------------------------------------------------
# SECCIÓN 6: Dashboard - Feature Importance y EDA
# ----------------------------------------------------------------------------------------------------------------------

        # ========================================================================
        # SECCIÓN 3: FEATURE IMPORTANCE
        # ========================================================================

        st.subheader(":material/sort: Feature Importance")
        st.markdown("**Based on Random Forest Feature Selection**")
        st.markdown("These are the feature importances used for selecting the top 12 features.")

        # Cargar feature importances (son las mismas para todos los modelos)
        if 'feature_importances' in feature_importance_data:
            fi_full = feature_importance_data['feature_importances'].copy()

            # Obtener features según la versión seleccionada
            if "Selected" in model_version:
                # Mostrar solo las 12 features seleccionadas
                selected_features = feature_config['selected_features']
                fi_df = fi_full[fi_full['Feature'].isin(selected_features)].copy()
                fi_df = fi_df.sort_values('Importance', ascending=True)
                title_suffix = "Selected Features (12)"
            else:
                # Mostrar top 15 de todas las features
                fi_df = fi_full.nlargest(15, 'Importance').sort_values('Importance', ascending=True)
                title_suffix = "All Features (Top 15)"

            # Crear gráfico de barras horizontal
            fig_fi = go.Figure(go.Bar(
                x=fi_df['Importance'],
                y=fi_df['Feature'],
                orientation='h',
                marker_color='#00CC96',
                text=[f"{val:.4f}" for val in fi_df['Importance']],
                textposition='auto',
                textfont=dict(size=18)
            ))

            fig_fi.update_layout(
                title=dict(
                    text=f"{title_suffix}",
                    font=dict(size=20)  # ← Tamaño del título
                ),
                xaxis=dict(
                    title=dict(
                        text="Importance Score",
                        font=dict(size=28)  # Tamaño del título del eje X
                    ),
                    tickfont=dict(size=18)  # Tamaño de los números del eje X
                ),
                yaxis=dict(
                    title=dict(
                        text="Feature",
                        font=dict(size=28)  # Tamaño del título del eje Y
                    ),
                    tickfont=dict(size=18)  # Tamaño de los nombres de features
                ),
                height=777,
                showlegend=False
            )

            st.plotly_chart(fig_fi, width='stretch')

            # Tabla completa en expander
            with st.expander(":material/format_list_bulleted: View All Feature Importances"):
                fi_table_full = fi_full.sort_values('Importance', ascending=False).copy()
                fi_table_full['Importance'] = fi_table_full['Importance'].apply(lambda x: f"{x:.6f}")
                fi_table_full['Cumulative_Importance'] = fi_table_full['Cumulative_Importance'].apply(
                    lambda x: f"{x:.4f}")
                st.dataframe(fi_table_full, width='stretch', hide_index=True)

                st.info(
                    "💡 **Note:** These importances were calculated using a Random Forest model and were used to select the top 12 features for the optimized models.")
        else:
            st.error("❌ Feature importance data not found in the loaded file.")

        st.markdown("---")
#         --------------------------------------------------------------------------------------------------------------
        # ========================================================================
        # COMPARACIÓN ENTRE LOS 3 MODELOS (DINÁMICO)
        # ========================================================================

        st.subheader(":material/leaderboard: Performance Comparison Across All Models")
        # st.markdown(f"**Comparing models using:** {model_version}")

        # Determinar qué métricas usar según la versión seleccionada
        if "Selected" in model_version:
            # Versión Selected Features
            lr_metrics = metrics_dict['lr_opt']
            stacking_metrics = metrics_dict['stacking_opt']
            voting_metrics = metrics_dict['voting_opt']
        else:
            # Versión All Features
            lr_metrics = metrics_dict['lr_all']
            stacking_metrics = metrics_dict['stacking_all']
            voting_metrics = metrics_dict['voting_all']

        # Crear gráfico de barras agrupadas (métricas en X, modelos por color)
        fig_models_comparison = go.Figure()

        metrics_names = ['Accuracy', 'AUC', 'F1-Score']

        # Logistic Regression (Verde)
        fig_models_comparison.add_trace(go.Bar(
            name='Logistic Regression',
            x=metrics_names,
            y=[lr_metrics['acc'], lr_metrics['auc'], lr_metrics['f1']],
            marker_color='#2ECC71',
            text=[f"{lr_metrics['acc']:.4f}", f"{lr_metrics['auc']:.4f}", f"{lr_metrics['f1']:.4f}"],
            textposition='auto',
            textfont=dict(size=16)
        ))

        # Stacking Classifier (Azul)
        fig_models_comparison.add_trace(go.Bar(
            name='Stacking Classifier',
            x=metrics_names,
            y=[stacking_metrics['acc'], stacking_metrics['auc'], stacking_metrics['f1']],
            marker_color='#3498DB',
            text=[f"{stacking_metrics['acc']:.4f}", f"{stacking_metrics['auc']:.4f}", f"{stacking_metrics['f1']:.4f}"],
            textposition='auto',
            textfont=dict(size=16)
        ))

        # Voting Classifier (Morado)
        fig_models_comparison.add_trace(go.Bar(
            name='Voting Classifier',
            x=metrics_names,
            y=[voting_metrics['acc'], voting_metrics['auc'], voting_metrics['f1']],
            marker_color='#9B59B6',
            text=[f"{voting_metrics['acc']:.4f}", f"{voting_metrics['auc']:.4f}", f"{voting_metrics['f1']:.4f}"],
            textposition='auto',
            textfont=dict(size=16)
        ))

        fig_models_comparison.update_layout(
            title=dict(
                text=f"Model Performance Comparison - {model_version}",
                font=dict(size=20)
            ),
            xaxis=dict(
                title=dict(text="Metric", font=dict(size=18)),
                tickfont=dict(size=16)
            ),
            yaxis=dict(
                title=dict(text="Score", font=dict(size=18)),
                tickfont=dict(size=16),
                range=[0, 1.05]
            ),
            barmode='group',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=16)
            )
        )

        st.plotly_chart(fig_models_comparison, width='stretch')

        # Tabla comparativa
        with st.expander(":material/table_view: View Detailed Comparison Table"):
            comparison_data = {
                'Model': ['Logistic Regression', 'Stacking Classifier', 'Voting Classifier'],
                'Accuracy': [
                    f"{lr_metrics['acc']:.4f}",
                    f"{stacking_metrics['acc']:.4f}",
                    f"{voting_metrics['acc']:.4f}"
                ],
                'AUC': [
                    f"{lr_metrics['auc']:.4f}",
                    f"{stacking_metrics['auc']:.4f}",
                    f"{voting_metrics['auc']:.4f}"
                ],
                'F1-Score': [
                    f"{lr_metrics['f1']:.4f}",
                    f"{stacking_metrics['f1']:.4f}",
                    f"{voting_metrics['f1']:.4f}"
                ]
            }

            df_comparison_display = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison_display, width='stretch', hide_index=True)

            # Identificar el mejor modelo por AUC
            auc_values = [lr_metrics['auc'], stacking_metrics['auc'], voting_metrics['auc']]
            best_model_idx = auc_values.index(max(auc_values))
            best_model_name = comparison_data['Model'][best_model_idx]
            best_auc = max(auc_values)

            st.success(f":material/looks_one: **Best Model (by AUC):** {best_model_name} with AUC = {best_auc:.4f}")

        st.markdown("---")
# ----------------------------------------------------------------------------------------------------------------------



elif selected_tab == 'EDA':
    with st.container(border=True):
        # ========================================================================
        # SECCIÓN 4: EXPLORATORY DATA ANALYSIS (EDA)
        # ========================================================================

        st.subheader(":material/browse: Dataset Exploration")

        # # Cargar datos preparados
        # X_train = prepared_data['X_train']
        # y_train = prepared_data['y_train']
        #
        # # Recrear dataset completo para EDA
        # df_eda = X_train.copy()
        # df_eda['Churn'] = y_train

        # Preparar datos para EDA con valores originales
        y_train = prepared_data['y_train']
        train_indices = prepared_data['X_train'].index

        df_eda = df_raw_for_eda.loc[train_indices].copy()
        df_eda['Churn'] = y_train.values

        # Sub-sección 1: Distribución de Churn
        st.markdown("##### :material/donut_large: Churn Distribution")

        eda_col1, eda_col2 = st.columns([2, 1])

        with eda_col1:
            # Gráfico de pie
            churn_counts = df_eda['Churn'].value_counts()

            fig_churn = go.Figure(data=[go.Pie(
                labels=['No Churn', 'Churn'],
                values=[churn_counts.get('No', 0), churn_counts.get('Yes', 0)],
                marker_colors=['#00CC96', '#EF553B'],
                hole=0.4,
                textinfo='label+percent',
                textfont_size=28
            )])

            fig_churn.update_layout(
                title="Target Variable Distribution",
                height=541,
                showlegend=True
            )

            st.plotly_chart(fig_churn, width='stretch')

        with eda_col2:
            st.markdown("**Class Distribution:**")
            st.metric("Total Customers", len(df_eda))
            st.metric("No Churn", churn_counts.get('No', 0))
            st.metric("Churn", churn_counts.get('Yes', 0))

            # churn_rate = (churn_counts.get('Yes', 0) / len(df_eda)) * 100
            # st.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.markdown("---")

        # Sub-sección 2: Distribuciones de Variables Numéricas
        st.markdown("#### :material/bar_chart: Numerical Features Distribution")

        # Seleccionar features numéricas
        numeric_features = ['tenure', 'MonthlyCharges']

        fig_hist = go.Figure()

        for feature in numeric_features:
            if feature in df_eda.columns:
                fig_hist.add_trace(go.Histogram(
                    x=df_eda[feature],
                    name=feature,
                    opacity=0.7,
                    nbinsx=30
                ))

        fig_hist.update_layout(
            title=dict(
                text="Distribution of Numerical Features",
                font=dict(size=20)  # ← Tamaño del título
            ),
            xaxis=dict(
                title=dict(text="Value", font=dict(size=18)),  # ← Título eje X
                tickfont=dict(size=14)  # ← Labels eje X
            ),
            yaxis=dict(
                title=dict(text="Frequency", font=dict(size=18)),  # ← Título eje Y
                tickfont=dict(size=14)  # ← Labels eje Y
            ),
            barmode='overlay',
            height=777,
            showlegend=True,
            legend=dict(
                font=dict(size=14)  # ← Tamaño texto de leyenda
            )
        )

        st.plotly_chart(fig_hist, width='stretch')

        st.markdown("---")

        # ----------------------------------------------------------------------------------------------------------------------

        # Sub-sección 3: Churn por características clave
        st.markdown("#### :material/query_stats: Churn Analysis by Key Features")

        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:
            # Churn por Contract
            if 'Contract' in df_eda.columns:
                churn_by_contract = pd.crosstab(df_eda['Contract'], df_eda['Churn'], normalize='index') * 100

                fig_contract = go.Figure()

                for churn_val in churn_by_contract.columns:
                    fig_contract.add_trace(go.Bar(
                        name=f'Churn: {churn_val}',
                        x=churn_by_contract.index,
                        y=churn_by_contract[churn_val],
                        text=[f"{val:.1f}%" for val in churn_by_contract[churn_val]],
                        textposition='auto',
                        textfont=dict(size=14)
                    ))

                fig_contract.update_layout(
                    title=dict(
                        text="Churn Rate by Contract Type",
                        font=dict(size=20)  # ← Tamaño del título
                    ),
                    xaxis=dict(
                        title=dict(text="Contract Type", font=dict(size=18)),  # ← Título eje X
                        tickfont=dict(size=14)  # ← Labels eje X
                    ),
                    yaxis=dict(
                        title=dict(text="Percentage (%)", font=dict(size=18)),  # ← Título eje Y
                        tickfont=dict(size=14),  # ← Labels eje Y
                        range=[0, 120]
                    ),
                    barmode='stack',
                    height=441,
                    legend=dict(
                        font=dict(size=14)  # ← Tamaño leyenda
                    )
                )

                st.plotly_chart(fig_contract, width='stretch')

        with analysis_col2:
            # Churn por Internet Service
            if 'InternetService' in df_eda.columns:
                churn_by_internet = pd.crosstab(df_eda['InternetService'], df_eda['Churn'], normalize='index') * 100

                fig_internet = go.Figure()

                for churn_val in churn_by_internet.columns:
                    fig_internet.add_trace(go.Bar(
                        name=f'Churn: {churn_val}',
                        x=churn_by_internet.index,
                        y=churn_by_internet[churn_val],
                        text=[f"{val:.1f}%" for val in churn_by_internet[churn_val]],
                        textposition='auto',
                        textfont=dict(size=14)
                    ))

                fig_internet.update_layout(
                    title=dict(
                        text="Churn Rate by Internet Service",
                        font=dict(size=20)  # ← Tamaño del título
                    ),
                    xaxis=dict(
                        title=dict(text="Internet Service Type", font=dict(size=18)),  # ← Título eje X
                        tickfont=dict(size=14)  # ← Labels eje X
                    ),
                    yaxis=dict(
                        title=dict(text="Percentage (%)", font=dict(size=18)),  # ← Título eje Y
                        tickfont=dict(size=14),  # ← Labels eje Y
                        range=[0, 120]
                    ),
                    barmode='stack',
                    height=441,
                    legend=dict(
                        font=dict(size=14)  # ← Tamaño leyenda
                    )

                )

                st.plotly_chart(fig_internet, width='stretch')

        st.markdown("---")

        # ----------------------------------------------------------------------------------------------------------------------

        # Sub-sección 4: Correlación entre todas las features preprocesadas
        st.markdown("#### :material/link: Correlation Analysis")

        # Usar datos PREPROCESADOS (las 22 features numéricas)
        df_corr = prepared_data['X_train'].copy()

        # Agregar Churn como numérica
        df_corr['Churn_Numeric'] = (prepared_data['y_train'] == 'Yes').astype(int)

        # Calcular matriz de correlación completa
        corr_matrix = df_corr.corr()

        # Ordenar por correlación con Churn (descendente)
        churn_corr = corr_matrix['Churn_Numeric'].sort_values(ascending=False)
        ordered_features = churn_corr.index.tolist()

        # Reordenar matriz según correlación con Churn
        corr_matrix_ordered = corr_matrix.loc[ordered_features, ordered_features]

        # Crear heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix_ordered.values,
            x=corr_matrix_ordered.columns,
            y=corr_matrix_ordered.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix_ordered.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(
                title=dict(text="Correlation", font=dict(size=14)),  # ← Título colorbar
                tickfont=dict(size=12)  # ← Labels colorbar
            )
        ))

        fig_corr.update_layout(
            title=dict(
                text="Correlation Matrix - All Preprocessed Features",
                font=dict(size=20)  # ← Tamaño del título
            ),
            height=800,
            xaxis=dict(
                side='bottom',
                tickangle=45,
                tickfont=dict(size=14)  # ← Labels eje X
            ),
            yaxis=dict(
                tickfont=dict(size=14)  # ← Labels eje Y
            )
        )

        st.plotly_chart(fig_corr, width='stretch')

        st.markdown("---")
        st.success(
            ":material/check_circle: Dashboard complete! Use the sidebar to explore different models and versions.")
