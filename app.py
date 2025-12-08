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