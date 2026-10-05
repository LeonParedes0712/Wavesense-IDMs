"""Runtime local EEG: características calculadas -> probabilidades originales."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ModelContractError(ValueError):
    """El artefacto o la entrada no cumple el contrato de inferencia."""


def _names(value):
    names = tuple(value)
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise ModelContractError("Los nombres de características deben ser textos no vacíos.")
    if len(set(names)) != len(names):
        raise ModelContractError("Los nombres de características no deben repetirse.")
    return names


def _matrix(value, label, *, allow_vector=False):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError("Se requieren números reales.")
        array = array.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelContractError(f"{label}: se requieren características numéricas reales.") from exc
    if allow_vector and array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ModelContractError(f"{label}: matriz no vacía, de dos dimensiones y sin NaN/inf requerida.")
    return array


class EEGModelRuntime:
    """Carga una vez y reutiliza. No adquiere EEG ni calcula características.

    El esquema procede de feature_names_in_ o de feature_names explícitos
    confirmados por entrenamiento. Los nombres de entrada son obligatorios.
    Modelo y escalador deben usar las mismas variables en el mismo orden.
    """

    def __init__(self, model, scaler=None, *, feature_names=None):
        self.model = model
        self.scaler = scaler
        if not callable(getattr(model, "predict_proba", None)):
            raise ModelContractError("El modelo debe implementar predict_proba().")
        classes = np.asarray(getattr(model, "classes_", []))
        if (classes.ndim != 1 or not classes.size
                or any(not isinstance(c, str) or not c for c in classes)
                or len(set(classes)) != len(classes)):
            raise ModelContractError("classes_ debe contener etiquetas de texto únicas del modelo.")
        self.classes = tuple(classes.tolist())
        if scaler is not None and not callable(getattr(scaler, "transform", None)):
            raise ModelContractError("El escalador debe implementar transform().")

        schemas = []
        if feature_names is not None:
            schemas.append(_names(feature_names))
        for estimator in (scaler, model):
            if hasattr(estimator, "feature_names_in_"):
                schemas.append(_names(estimator.feature_names_in_))
        if not schemas:
            raise ModelContractError(
                "Falta el contrato de características: exportar feature_names_in_ "
                "o proporcionar feature_names con nombres y orden confirmados por entrenamiento; "
                "n_features_in_ por sí solo no identifica las variables."
            )
        if any(schema != schemas[0] for schema in schemas):
            raise ModelContractError("Modelo, escalador y contrato deben compartir nombres y orden.")
        self.feature_names = schemas[0]
        self.n_features = len(self.feature_names)
        for estimator in (scaler, model):
            count = getattr(estimator, "n_features_in_", self.n_features)
            if count != self.n_features:
                raise ModelContractError("n_features_in_ no coincide con el contrato de características.")

    @classmethod
    def from_env(cls, *, feature_names=None):
        """Lee EEG_MODEL_PATH (obligatoria) y EEG_SCALER_PATH (opcional).

        Rutas relativas al directorio de trabajo. Solo artefactos joblib/pickle
        locales de confianza; deserializar pickle puede ejecutar código.
        La aplicación, si lo necesita, carga .env antes de llamar este método.
        """
        model_path = os.environ.get("EEG_MODEL_PATH", "").strip()
        scaler_path = os.environ.get("EEG_SCALER_PATH", "").strip()
        if not model_path:
            raise ModelContractError("Falta EEG_MODEL_PATH: configura la ruta al modelo local.")

        def load(path, variable):
            path = Path(path).expanduser()
            if not path.is_file():
                raise ModelContractError(f"{variable}: no existe un archivo local en {path}.")
            try:
                return joblib.load(path)
            except Exception as exc:
                raise ModelContractError(
                    f"{variable}: no se pudo cargar {path}; verifica formato y versiones de entrenamiento."
                ) from exc

        model = load(model_path, "EEG_MODEL_PATH")
        scaler = load(scaler_path, "EEG_SCALER_PATH") if scaler_path else None
        return cls(model, scaler, feature_names=feature_names)

    def _input(self, features, feature_names):
        if isinstance(features, pd.DataFrame):
            names = _names(features.columns)
            if feature_names is not None and _names(feature_names) != names:
                raise ModelContractError("feature_names no coincide con las columnas del DataFrame.")
        elif feature_names is None:
            raise ModelContractError(
                "Faltan nombres y orden de entrada: entrega un DataFrame o feature_names explícitos."
            )
        else:
            names = _names(feature_names)
        array = _matrix(features, "Entrada", allow_vector=True)
        if array.shape[1] != self.n_features:
            raise ModelContractError(
                f"Se esperaban {self.n_features} variables; se recibieron {array.shape[1]}."
            )
        if names != self.feature_names:
            raise ModelContractError(
                f"Nombres u orden incorrectos. Se esperaba: {self.feature_names}."
            )
        return array

    @staticmethod
    def _for_estimator(array, estimator, names):
        if hasattr(estimator, "feature_names_in_"):
            return pd.DataFrame(array, columns=names)
        return array

    def _predict(self, array):
        if self.scaler is not None:
            scaled = self.scaler.transform(
                self._for_estimator(array, self.scaler, self.feature_names)
            )
            if isinstance(scaled, pd.DataFrame) and tuple(scaled.columns) != self.feature_names:
                raise ModelContractError("El escalador cambió los nombres u orden de columnas.")
            scaled = _matrix(scaled, "Salida del escalador")
            if scaled.shape != array.shape:
                raise ModelContractError("El escalador cambió la forma de la matriz de características.")
            array = scaled
        probabilities = _matrix(self.model.predict_proba(
            self._for_estimator(array, self.model, self.feature_names)
        ), "predict_proba")
        if probabilities.shape != (len(array), len(self.classes)):
            raise ModelContractError("predict_proba no coincide con las filas de entrada y classes_.")
        if ((probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)):
            raise ModelContractError("predict_proba debe devolver probabilidades en [0, 1] que sumen 1.")
        return [dict(zip(self.classes, map(float, row))) for row in probabilities]

    def predict_proba(self, features, *, feature_names=None) -> dict[str, float]:
        """Vector (p,) o matriz (1,p) -> dict directamente compatible con el adaptador."""
        array = self._input(features, feature_names)
        if len(array) != 1:
            raise ModelContractError("Para varias ventanas usa predict_proba_batch().")
        return self._predict(array)[0]

    def predict_proba_batch(self, features, *, feature_names=None) -> list[dict[str, float]]:
        """Matriz (n,p) -> un diccionario por ventana, conservando el orden de filas."""
        return self._predict(self._input(features, feature_names))
