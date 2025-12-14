"""
utils.py
Custom Transformer Classes for Telco Churn Prediction Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ======================================================================================================================
# CUSTOM TRANSFORMERS
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 1. CustomerID Dropper
# ----------------------------------------------------------------------------------------------------------------------
class CustomerIDDropper(BaseEstimator, TransformerMixin):
    """Elimina la columna customerID si existe"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if 'customerID' in X_copy.columns:
            X_copy = X_copy.drop('customerID', axis=1)
        return X_copy

# ----------------------------------------------------------------------------------------------------------------------
# 2. TotalCharges Cleaner
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# 3. TotalCharges Dropper
# ----------------------------------------------------------------------------------------------------------------------
class TotalChargesDropper(BaseEstimator, TransformerMixin):
    """Elimina TotalCharges (multicolinealidad con tenure)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if 'TotalCharges' in X_copy.columns:
            X_copy = X_copy.drop('TotalCharges', axis=1)
        return X_copy

# ----------------------------------------------------------------------------------------------------------------------
# 4. Categorical Redundancy Cleaner
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# 5. Custom OneHotEncoder (con drop='first')
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# 6. Complete Preprocessing Pipeline
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# 7. DataFrame Preparer (alias/wrapper de CompletePreprocessingPipeline)
# ----------------------------------------------------------------------------------------------------------------------
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

