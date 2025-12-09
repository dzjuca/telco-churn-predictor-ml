# ============================================================================
# TELCO CHURN PREDICTOR - STREAMLIT APP
# Machine Learning Final Exam - Master's in Data Science
# ============================================================================

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

# ============================================================================
# CLASES PERSONALIZADAS (necesarias para cargar el pipeline)
# ============================================================================

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ----------------------------------------------------------------------------
# 1. CustomerID Dropper
# ----------------------------------------------------------------------------
class CustomerIDDropper(BaseEstimator, TransformerMixin):
    """Elimina la columna customerID si existe"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if 'customerID' in X_copy.columns:
            X_copy = X_copy.drop('customerID', axis=1)
        return X_copy


# ----------------------------------------------------------------------------
# 2. TotalCharges Cleaner
# ----------------------------------------------------------------------------
class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """Limpia TotalCharges: espacios → '0', convierte a float"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if 'TotalCharges' in X_copy.columns:
            X_copy['TotalCharges'] = X_copy['TotalCharges'].replace(' ', '0')
            X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'])
        return X_copy


# ----------------------------------------------------------------------------
# 3. TotalCharges Dropper
# ----------------------------------------------------------------------------
class TotalChargesDropper(BaseEstimator, TransformerMixin):
    """Elimina TotalCharges (multicolinealidad con tenure)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if 'TotalCharges' in X_copy.columns:
            X_copy = X_copy.drop('TotalCharges', axis=1)
        return X_copy


# ----------------------------------------------------------------------------
# 4. Categorical Redundancy Cleaner
# ----------------------------------------------------------------------------
class CategoricalRedundancyCleaner(BaseEstimator, TransformerMixin):
    """
    Limpia redundancias:
    - 'No internet service' → 'No'
    - 'No phone service' → 'No'
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        # Features con 'No internet service'
        internet_features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                             'TechSupport', 'StreamingTV', 'StreamingMovies']
        for col in internet_features:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace('No internet service', 'No')

        # Feature con 'No phone service'
        if 'MultipleLines' in X_copy.columns:
            X_copy['MultipleLines'] = X_copy['MultipleLines'].replace('No phone service', 'No')

        return X_copy


# ----------------------------------------------------------------------------
# 5. Custom OneHotEncoder (con drop='first')
# ----------------------------------------------------------------------------
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """OneHotEncoder con drop='first' para evitar dummy variable trap"""

    def __init__(self):
        try:
            self._oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first')
        except TypeError:
            self._oh = OneHotEncoder(sparse=False, handle_unknown="ignore", drop='first')
        self._columns = []

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object']).copy()
        if X_cat.shape[1] == 0:
            self._columns = []
            self._oh.fit(pd.DataFrame(index=X.index))
            return self

        self._oh.fit(X_cat)
        self._columns = self._oh.get_feature_names_out(X_cat.columns)
        return self

    def transform(self, X, y=None):
        X_cat = X.select_dtypes(include=['object']).copy()
        if X_cat.shape[1] == 0:
            return pd.DataFrame(index=X.index)

        X_cat_oh = self._oh.transform(X_cat)
        return pd.DataFrame(X_cat_oh, columns=self._columns, index=X.index)


# ----------------------------------------------------------------------------
# 6. Complete Preprocessing Pipeline
# ----------------------------------------------------------------------------
class CompletePreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Pipeline COMPLETO que aplica TODAS las transformaciones en orden:
    1. Eliminar customerID
    2. Limpiar TotalCharges
    3. Eliminar TotalCharges (multicolinealidad)
    4. Limpiar redundancias categóricas
    5. Encoding (OneHot con drop='first') + Scaling
    """

    def __init__(self):
        self._customer_dropper = CustomerIDDropper()
        self._totalcharges_cleaner = TotalChargesCleaner()
        self._totalcharges_dropper = TotalChargesDropper()
        self._redundancy_cleaner = CategoricalRedundancyCleaner()
        self._full_pipeline = None
        self._columns = None
        self.input_features_ = None

    def fit(self, X, y=None):
        X1 = self._customer_dropper.fit_transform(X)
        X2 = self._totalcharges_cleaner.fit_transform(X1)
        X3 = self._totalcharges_dropper.fit_transform(X2)
        X4 = self._redundancy_cleaner.fit_transform(X3)

        self.input_features_ = list(X4.columns)

        num_attribs = list(X4.select_dtypes(exclude=['object']).columns)
        cat_attribs = list(X4.select_dtypes(include=['object']).columns)

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('scaler', StandardScaler()),
        ])

        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", CustomOneHotEncoder(), cat_attribs),
        ])

        self._full_pipeline.fit(X4)

        out_cols = []
        out_cols.extend(num_attribs)
        cat_encoder = self._full_pipeline.named_transformers_["cat"]
        if hasattr(cat_encoder, "_columns") and len(cat_encoder._columns) > 0:
            out_cols.extend(list(cat_encoder._columns))

        self._columns = out_cols
        return self

    def transform(self, X, y=None):
        X1 = self._customer_dropper.transform(X)
        X2 = self._totalcharges_cleaner.transform(X1)
        X3 = self._totalcharges_dropper.transform(X2)
        X4 = self._redundancy_cleaner.transform(X3)

        X_prep = self._full_pipeline.transform(X4)

        return pd.DataFrame(X_prep, columns=self._columns, index=X.index)


# ----------------------------------------------------------------------------
# 7. DataFrame Preparer (alias/wrapper de CompletePreprocessingPipeline)
# ----------------------------------------------------------------------------
class DataFramePreparer(BaseEstimator, TransformerMixin):
    """
    Wrapper/alias de CompletePreprocessingPipeline
    (por compatibilidad con notebooks anteriores)
    """

    def __init__(self):
        self._pipeline = CompletePreprocessingPipeline()

    def fit(self, X, y=None):
        self._pipeline.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self._pipeline.transform(X, y)

    @property
    def _columns(self):
        return self._pipeline._columns

    @property
    def input_features_(self):
        return self._pipeline.input_features_


# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# TÍTULO PRINCIPAL
# ============================================================================

st.title("📊 Telco Customer Churn Prediction")
st.markdown("---")
st.markdown("""
### Machine Learning Ensemble Models Dashboard
This application allows you to predict customer churn using three different ensemble models,
compare their performance, and explore the dataset.
""")

st.markdown("---")
st.success("✅ App initialized successfully!")


# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================

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


# ============================================================================
# CARGAR TODOS LOS DATOS AL INICIO
# ============================================================================

try:
    models_dict = load_models()
    preprocessing_pipeline = load_pipeline()
    feature_importance_data = load_feature_importance()
    metrics_dict = load_metrics()
    feature_config = load_feature_config()
    summary_table = load_summary_table()
    prepared_data = load_prepared_data()

    st.success("✅ All models and data loaded successfully!")

    # Mostrar información de debug (temporal)
    with st.expander("🔍 Debug Info - Click to expand"):
        st.write("**Models loaded:**", list(models_dict.keys()))
        st.write("**Metrics available:**", list(metrics_dict.keys()))
        st.write("**Feature config keys:**", list(feature_config.keys()))

except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()


# ============================================================================
# SIDEBAR - SELECTORES
# ============================================================================

st.sidebar.title("🎯 Model Selection")
st.sidebar.markdown("---")

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

st.sidebar.markdown("---")

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
st.sidebar.success(f"✅ Model loaded: **{model_type}**")
st.sidebar.info(f"📊 Version: **{model_version}**")

# Mostrar métricas del modelo seleccionado
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Performance")
st.sidebar.metric("Accuracy", f"{selected_metrics['acc']:.4f}")
st.sidebar.metric("AUC", f"{selected_metrics['auc']:.4f}")
st.sidebar.metric("F1-Score", f"{selected_metrics['f1']:.4f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tip:** Use the tabs below to make predictions or explore model performance.")

# ============================================================================
# TABS PRINCIPALES
# ============================================================================

tab1, tab2 = st.tabs(["🔮 Make Prediction", "📊 Model Dashboard"])

# ============================================================================
# TAB 1: FORMULARIO DE INFERENCIA
# ============================================================================

with tab1:
    st.header("🔮 Customer Churn Prediction")
    st.markdown("Enter customer information to predict churn probability.")
    st.markdown("---")

    # Crear formulario en 3 columnas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Customer Info")

        # Campo calculado: Customer ID
        st.text_input("Customer ID", value="PRED-001", disabled=True,
                      help="Auto-generated ID for prediction")

        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

        st.subheader("📞 Phone Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col2:
        st.subheader("🌐 Internet Services")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("💳 Account Info")
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
    if st.button("🚀 Predict Churn", type="primary", width='stretch'):

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
            st.subheader("🎯 Prediction Results")

            # Crear 2 columnas para los resultados
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if prediction == "Yes":
                    st.error("### ⚠️ CHURN RISK: HIGH")
                    st.markdown("**Prediction:** Customer is likely to churn")
                else:
                    st.success("### ✅ CHURN RISK: LOW")
                    st.markdown("**Prediction:** Customer is likely to stay")

            with result_col2:
                churn_prob = prediction_proba[1] if hasattr(prediction_proba, '__len__') else prediction_proba
                st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")
                st.metric("Retention Probability", f"{(1 - churn_prob) * 100:.2f}%")

            # Mostrar probabilidades como barra de progreso
            st.markdown("---")
            st.markdown("**Probability Distribution:**")
            st.progress(float(churn_prob), text=f"Churn: {churn_prob * 100:.1f}%")

            # Información adicional
            with st.expander("📋 View Input Data"):
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

with tab2:
    st.header("📊 Model Performance Dashboard")
    st.markdown("Explore model metrics, confusion matrices, feature importance, and dataset insights.")
    st.markdown("---")

    # ========================================================================
    # SECCIÓN 1: COMPARACIÓN DE MÉTRICAS
    # ========================================================================

    st.subheader("📈 Performance Metrics Comparison")
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
        marker_color='#636EFA',
        text=[f"{val:.4f}" for val in comparison_df['All Features (22)']],
        textposition='auto',
    ))

    fig_comparison.add_trace(go.Bar(
        name='Selected Features (12)',
        x=comparison_df['Metric'],
        y=comparison_df['Selected Features (12)'],
        marker_color='#EF553B',
        text=[f"{val:.4f}" for val in comparison_df['Selected Features (12)']],
        textposition='auto',
    ))

    fig_comparison.update_layout(
        title=f"{model_name_short}: All Features vs Selected Features",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode='group',
        yaxis=dict(range=[0, 1.05]),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_comparison, width='stretch')

    # Mostrar tabla comparativa
    with st.expander("📋 View Detailed Metrics Table"):
        # Formatear tabla
        comparison_display = comparison_df.copy()
        comparison_display['All Features (22)'] = comparison_display['All Features (22)'].apply(lambda x: f"{x:.4f}")
        comparison_display['Selected Features (12)'] = comparison_display['Selected Features (12)'].apply(
            lambda x: f"{x:.4f}")
        st.dataframe(comparison_display, width='stretch', hide_index=True)

    st.markdown("---")

    # ========================================================================
    # SECCIÓN 2: CONFUSION MATRICES
    # ========================================================================

    st.subheader("🔥 Confusion Matrices")

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
            textfont={"size": 16},
            colorscale='Blues',
            showscale=False
        ))

        fig_cm_all.update_layout(
            title=f"{model_name_short} - All Features",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=350,
            yaxis=dict(autorange='reversed')
        )

        st.plotly_chart(fig_cm_all, width='stretch')

        # Métricas derivadas
        tn, fp, fn, tp = cm_all.ravel()
        st.metric("True Negatives (TN)", tn)
        st.metric("True Positives (TP)", tp)
        st.metric("False Positives (FP)", fp)
        st.metric("False Negatives (FN)", fn)

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
            textfont={"size": 16},
            colorscale='Reds',
            showscale=False
        ))

        fig_cm_opt.update_layout(
            title=f"{model_name_short} - Selected Features",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=350,
            yaxis=dict(autorange='reversed')
        )

        st.plotly_chart(fig_cm_opt, width='stretch')

        # Métricas derivadas
        tn, fp, fn, tp = cm_opt.ravel()
        st.metric("True Negatives (TN)", tn)
        st.metric("True Positives (TP)", tp)
        st.metric("False Positives (FP)", fp)
        st.metric("False Negatives (FN)", fn)

    st.markdown("---")


# ----------------------------------------------------------------------------------------------------------------------
# SECCIÓN 6: Dashboard - Feature Importance y EDA
# ----------------------------------------------------------------------------------------------------------------------

    # ========================================================================
    # SECCIÓN 3: FEATURE IMPORTANCE
    # ========================================================================

    st.subheader("⭐ Feature Importance")
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
        ))

        fig_fi.update_layout(
            title=f"Feature Importance - {title_suffix}",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig_fi, width='stretch')

        # Tabla completa en expander
        with st.expander("📋 View All Feature Importances"):
            fi_table_full = fi_full.sort_values('Importance', ascending=False).copy()
            fi_table_full['Importance'] = fi_table_full['Importance'].apply(lambda x: f"{x:.6f}")
            fi_table_full['Cumulative_Importance'] = fi_table_full['Cumulative_Importance'].apply(lambda x: f"{x:.4f}")
            st.dataframe(fi_table_full, width='stretch', hide_index=True)

            st.info(
                "💡 **Note:** These importances were calculated using a Random Forest model and were used to select the top 12 features for the optimized models.")
    else:
        st.error("❌ Feature importance data not found in the loaded file.")

    st.markdown("---")


    # ========================================================================
    # SECCIÓN 4: EXPLORATORY DATA ANALYSIS (EDA)
    # ========================================================================

    st.subheader("📈 Dataset Exploration")

    # Cargar datos preparados
    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']

    # Recrear dataset completo para EDA
    df_eda = X_train.copy()
    df_eda['Churn'] = y_train

    # Sub-sección 1: Distribución de Churn
    st.markdown("#### 🎯 Churn Distribution")

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
            textfont_size=14
        )])

        fig_churn.update_layout(
            title="Target Variable Distribution",
            height=350,
            showlegend=True
        )

        st.plotly_chart(fig_churn, width='stretch')

    with eda_col2:
        st.markdown("**Class Distribution:**")
        st.metric("Total Customers", len(df_eda))
        st.metric("No Churn", churn_counts.get('No', 0))
        st.metric("Churn", churn_counts.get('Yes', 0))

        churn_rate = (churn_counts.get('Yes', 0) / len(df_eda)) * 100
        st.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.markdown("---")

    # Sub-sección 2: Distribuciones de Variables Numéricas
    st.markdown("#### 📊 Numerical Features Distribution")

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
        title="Distribution of Numerical Features",
        xaxis_title="Value",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig_hist, width='stretch')

    st.markdown("---")

    # Sub-sección 3: Churn por características clave
    st.markdown("#### 🔍 Churn Analysis by Key Features")

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
                ))

            fig_contract.update_layout(
                title="Churn Rate by Contract Type",
                xaxis_title="Contract Type",
                yaxis_title="Percentage (%)",
                barmode='stack',
                height=350
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
                ))

            fig_internet.update_layout(
                title="Churn Rate by Internet Service",
                xaxis_title="Internet Service Type",
                yaxis_title="Percentage (%)",
                barmode='stack',
                height=350
            )

            st.plotly_chart(fig_internet, width='stretch')

    st.markdown("---")

    # Sub-sección 4: Correlación entre variables numéricas y Churn
    st.markdown("#### 🔗 Correlation Analysis")

    # Convertir Churn a numérico para correlación
    df_corr = df_eda.copy()
    df_corr['Churn_Numeric'] = (df_corr['Churn'] == 'Yes').astype(int)

    # Seleccionar solo columnas numéricas
    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 1:
        # Calcular matriz de correlación
        corr_matrix = df_corr[numeric_cols].corr()

        # Crear heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig_corr.update_layout(
            title="Correlation Matrix - Numerical Features",
            height=500,
            xaxis={'side': 'bottom'},
        )

        st.plotly_chart(fig_corr, width='stretch')
    else:
        st.info("Not enough numerical features for correlation analysis.")

    st.markdown("---")
    st.success("✅ Dashboard complete! Use the sidebar to explore different models and versions.")