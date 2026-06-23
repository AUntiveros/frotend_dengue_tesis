"""
Cargadores e inferencia de cada familia de modelos.

Contrato uniforme: cada modelo distrital expone
    pred_raw(df) -> np.ndarray
con la salida nativa del modelo (ya invertida a escala de casos) para cada
fila de `df`. Para los modelos de horizonte (árboles, BiLSTM) esa salida es
la estimación de casos en `fecha + 4 semanas`; el orquestador la realinea.

El modelo ZIP es contemporáneo (estima casos en la misma `fecha`) y además
expone contribuciones locales β·x como importancia de variables real.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd

from . import comun

warnings.filterwarnings("ignore")

MOD = comun.MODELOS_DIR


# ════════════════════════════════════════════════════════════════════════════
# ZIP Bayesiano M7b (Horseshoe) — el mejor modelo, inferencia rápida en numpy
# ════════════════════════════════════════════════════════════════════════════
class ZipM7b:
    id = "M7b"
    forward = False          # contemporáneo
    family = "Bayesiano"

    def __init__(self):
        with open(MOD / "scaler_zip.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        self.feats = list(self.scaler.feature_names_in_)
        self.beta = np.load(MOD / "zip_beta_mean.npy").astype(float)
        self.intercept = float(np.load(MOD / "zip_intercept.npy")[0])
        # psi posterior por cluster LISA: [HH, LL, ns]
        self.psi = np.load(MOD / "zip_psi_posterior.npy").astype(float)

    def _psi_vec(self, cluster_code: np.ndarray) -> np.ndarray:
        return np.where(cluster_code == 4, self.psi[0],
               np.where(cluster_code == 1, self.psi[1], self.psi[2]))

    def _lambda(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[self.feats].fillna(0).values)
        return np.exp(np.clip(self.intercept + X @ self.beta, -10, 12)), X

    def pred_raw(self, df: pd.DataFrame) -> np.ndarray:
        lam, _ = self._lambda(df)
        psi = self._psi_vec(df["cluster_code"].values)
        return (1.0 - psi) * lam

    def pred_intervalo(self, df: pd.DataFrame):
        """E[Y] e IC95% bajo el modelo ZIP."""
        lam, _ = self._lambda(df)
        psi = self._psi_vec(df["cluster_code"].values)
        ey = (1.0 - psi) * lam
        var = (1.0 - psi) * lam * (1.0 + psi * lam)
        sd = np.sqrt(var)
        lo = np.clip(ey - 1.96 * sd, 0, None)
        hi = ey + 1.96 * sd
        return ey, lo, hi

    def shap_local(self, fila: pd.DataFrame, top: int = 8):
        """Contribución local β_j · x_scaled_j al log-λ (importancia real)."""
        X = self.scaler.transform(fila[self.feats].fillna(0).values)[0]
        contrib = self.beta * X
        orden = np.argsort(np.abs(contrib))[::-1][:top]
        m = np.abs(contrib[orden]).max() or 1.0
        return [[self.feats[i], float(round(contrib[i] / m, 3))] for i in orden]


# ════════════════════════════════════════════════════════════════════════════
# Árboles de decisión (XGBoost, LightGBM, Random Forest)
# ════════════════════════════════════════════════════════════════════════════
class ArbolModel:
    forward = True

    def __init__(self, id, family, feats, predict_fn, transform, importancia_fn):
        self.id = id
        self.family = family
        self.feats = feats
        self._predict = predict_fn
        self.transform = transform            # 'log' | 'poisson'
        self._importancia = importancia_fn

    def pred_raw(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._predict(df[self.feats].fillna(0).values.astype("float32"))
        raw = np.asarray(raw, dtype=float)
        if self.transform == "log":
            return np.expm1(np.clip(raw, 0, None))
        return np.clip(raw, 0, None)          # poisson / raw

    def shap_local(self, fila, top: int = 8):
        imp = self._importancia()
        orden = np.argsort(imp)[::-1][:top]
        m = imp[orden].max() or 1.0
        return [[self.feats[i], float(round(imp[i] / m, 3))] for i in orden]


def cargar_arboles(feats):
    """Devuelve los modelos basados en árboles que cargan correctamente."""
    import xgboost as xgb
    import lightgbm as lgb

    modelos = {}
    n_esperado = len(feats)

    def add(id, family, predict_fn, transform, imp_fn, nfeat):
        if nfeat != n_esperado:
            print(f"  ! {id} omitido: entrenado con {nfeat} features (se esperan {n_esperado})")
            return
        modelos[id] = ArbolModel(id, family, feats, predict_fn, transform, imp_fn)

    # XGBoost log1p
    try:
        m = xgb.XGBRegressor()
        m.load_model(str(MOD / "xgb_log.json"))
        add("XGB_LOG", "Gradient Boosting", m.predict, "log",
            lambda m=m: np.asarray(m.feature_importances_, float), m.n_features_in_)
    except Exception as e:
        print(f"  ! XGB_LOG omitido: {e}")

    # XGBoost Poisson
    try:
        m = xgb.XGBRegressor()
        m.load_model(str(MOD / "xgb_poisson.json"))
        add("XGB_POI", "Gradient Boosting", m.predict, "poisson",
            lambda m=m: np.asarray(m.feature_importances_, float), m.n_features_in_)
    except Exception as e:
        print(f"  ! XGB_POI omitido: {e}")

    # LightGBM Optuna
    try:
        b = lgb.Booster(model_file=str(MOD / "lgb_optuna.txt"))
        add("LGB_OPT", "Gradient Boosting", b.predict, "log",
            lambda b=b: np.asarray(b.feature_importance("gain"), float), b.num_feature())
    except Exception as e:
        print(f"  ! LGB_OPT omitido: {e}")

    # LightGBM global (entrenado con subconjunto reducido — puede no aplicar)
    try:
        b = lgb.Booster(model_file=str(MOD / "lgb_global.txt"))
        add("LGB_GLB", "Gradient Boosting", b.predict, "log",
            lambda b=b: np.asarray(b.feature_importance("gain"), float), b.num_feature())
    except Exception as e:
        print(f"  ! LGB_GLB omitido: {e}")

    # Random Forest
    try:
        with open(MOD / "rf_model.pkl", "rb") as f:
            rf = pickle.load(f)
        add("RF", "Bagging", rf.predict, "log",
            lambda rf=rf: np.asarray(rf.feature_importances_, float), rf.n_features_in_)
    except Exception as e:
        print(f"  ! RF omitido: {e}")

    return modelos


# ════════════════════════════════════════════════════════════════════════════
# Stacking — meta-modelo lineal positivo sobre [XGB_log, LGB_opt, RF] (en log)
# ════════════════════════════════════════════════════════════════════════════
class StackingModel:
    id = "STACK"
    forward = True
    family = "ML Ensamble"

    def __init__(self, feats):
        import xgboost as xgb
        import lightgbm as lgb
        self.feats = feats
        self.coef = np.load(MOD / "stack_coef.npy").astype(float)  # [xgb, lgb, rf]
        self.xgb = xgb.XGBRegressor(); self.xgb.load_model(str(MOD / "xgb_log.json"))
        self.lgb = lgb.Booster(model_file=str(MOD / "lgb_optuna.txt"))
        with open(MOD / "rf_model.pkl", "rb") as f:
            self.rf = pickle.load(f)

    def pred_raw(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feats].fillna(0).values.astype("float32")
        base_log = np.column_stack([
            self.xgb.predict(X),
            self.lgb.predict(X),
            self.rf.predict(X),
        ])
        pred_log = base_log @ self.coef          # combinación en espacio log
        return np.expm1(np.clip(pred_log, 0, None))

    def shap_local(self, fila, top: int = 8):
        # La "importancia" del stacking es el peso de cada modelo base
        pesos = self.coef / (np.abs(self.coef).max() or 1.0)
        nombres = ["XGBoost (log1p)", "LightGBM (Optuna)", "Random Forest"]
        pares = sorted(zip(nombres, pesos), key=lambda t: abs(t[1]), reverse=True)
        return [[n, float(round(w, 3))] for n, w in pares]


# ════════════════════════════════════════════════════════════════════════════
# BiLSTM (PyTorch) — secuencias de 12 semanas, target log1p
# ════════════════════════════════════════════════════════════════════════════
class BiLSTMModel:
    id = "BILSTM"
    forward = True
    family = "Deep Learning"

    def __init__(self, feats):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.feats = feats
        self.window = comun.LSTM_WINDOW
        with open(MOD / "scaler_lstm.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        n_feat = len(feats)

        class BiLSTMForecaster(nn.Module):
            def __init__(self, n_feat, h1=64, h2=32, dropout=0.3):
                super().__init__()
                self.bilstm1 = nn.LSTM(n_feat, h1, batch_first=True, bidirectional=True)
                self.dropout1 = nn.Dropout(dropout)
                self.bilstm2 = nn.LSTM(h1 * 2, h2, batch_first=True, bidirectional=True)
                self.dropout2 = nn.Dropout(dropout)
                self.bn = nn.BatchNorm1d(h2 * 2)
                self.head = nn.Sequential(nn.Linear(h2 * 2, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(self, x):
                out, _ = self.bilstm1(x)
                out = self.dropout1(out)
                out, _ = self.bilstm2(out)
                out = self.dropout2(out[:, -1, :])
                out = self.bn(out)
                return self.head(out).squeeze(-1)

        self.model = BiLSTMForecaster(n_feat)
        state = torch.load(MOD / "bilstm_best.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def pred_panel(self, df: pd.DataFrame) -> pd.Series:
        """
        Predicción secuencial por distrito. Devuelve una Serie indexada por
        el índice original de `df` (NaN donde no hay 12 semanas de contexto).
        Salida nativa = estimación de casos en fecha+4 (igual que los árboles).
        """
        torch = self.torch
        feats, W = self.feats, self.window
        out = pd.Series(np.nan, index=df.index, dtype=float)

        seqs, idxs = [], []
        for _, g in df.groupby("ubigeo", sort=False):
            g = g.sort_values("fecha")
            vals = self.scaler.transform(g[feats].fillna(0).values).astype("float32")
            gi = g.index.values
            for i in range(W, len(vals) + 1):
                seqs.append(vals[i - W:i])
                idxs.append(gi[i - 1])           # predicción asociada a la última semana
        if not seqs:
            return out

        X = torch.from_numpy(np.asarray(seqs, dtype="float32"))
        preds = []
        with torch.no_grad():
            for k in range(0, len(X), 4096):
                preds.append(self.model(X[k:k + 4096]).cpu().numpy())
        raw = np.concatenate(preds)
        out.loc[idxs] = np.expm1(np.clip(raw, 0, None))
        return out

    def shap_local(self, fila, top: int = 8):
        return None          # el BiLSTM no expone importancia nativa


# ════════════════════════════════════════════════════════════════════════════
# Prophet — modelo NACIONAL (serie agregada + regresores satelitales)
# ════════════════════════════════════════════════════════════════════════════
class ProphetNac:
    id = "PROPHET"
    family = "Series de tiempo"
    scope = "nacional"

    REGRESORES = ["ndvi_lag4", "precip_lag4", "lst_lag4", "tmean_x_ndvi"]

    def __init__(self):
        with open(MOD / "prophet_full.pkl", "rb") as f:
            self.model = pickle.load(f)

    def _serie_nacional(self, df: pd.DataFrame) -> pd.DataFrame:
        serie = (df[df["fecha"].notna()]
                 .groupby("fecha")
                 .agg(y=("casos", "sum"),
                      ndvi=("ndvi", "mean"),
                      precip_mm=("precip_chirps_mm", "mean"),
                      lst_norm=("seq_lst_norm", "mean"),
                      tmean=("tmean_clima", "mean"))
                 .reset_index()
                 .rename(columns={"fecha": "ds"}))
        serie["ndvi_lag4"] = serie["ndvi"].shift(4)
        serie["precip_lag4"] = serie["precip_mm"].shift(4)
        serie["lst_lag4"] = serie["lst_norm"].shift(4)
        serie["tmean_lag9"] = serie["tmean"].shift(9)
        serie["tmean_x_ndvi"] = serie["tmean_lag9"] * serie["ndvi_lag4"]
        return serie

    def predecir(self, df: pd.DataFrame, fechas) -> pd.DataFrame:
        serie = self._serie_nacional(df)
        sub = serie[serie["ds"].isin(fechas)][["ds"] + self.REGRESORES].copy()
        for c in self.REGRESORES:
            sub[c] = sub[c].fillna(serie[c].median())
        fc = self.model.predict(sub)
        fc["yhat"] = fc["yhat"].clip(lower=0)
        return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
