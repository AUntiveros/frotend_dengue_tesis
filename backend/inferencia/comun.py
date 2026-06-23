"""
Utilidades compartidas para la generación del dashboard.

Carga la AMT, el metadata y resuelve la semana de referencia (la última
semana epidemiológica con señal real). Centraliza constantes y helpers
numéricos usados por todos los modelos.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent          # .../backend
DATA_DIR    = BACKEND_DIR / "data"
MODELOS_DIR = BACKEND_DIR / "modelos"

AMT_PARQUET  = DATA_DIR / "AMT_final.parquet"
AMT_METADATA = DATA_DIR / "AMT_metadata.json"

# ── Parámetros del dashboard ─────────────────────────────────────────────────
N_HIST = 52          # semanas de historia mostradas en la serie temporal
LSTM_WINDOW = 12     # ventana de contexto del BiLSTM (debe coincidir con 03b)
HORIZONTE = 4        # única H con artefactos entrenados/guardados


def cargar_metadata() -> dict:
    with open(AMT_METADATA, encoding="utf-8") as f:
        return json.load(f)


def cargar_amt() -> pd.DataFrame:
    """AMT deduplicada por (ubigeo, fecha) y ordenada temporalmente."""
    df = pd.read_parquet(AMT_PARQUET)
    df = df.sort_values(["ubigeo", "fecha"])
    # Algunas semanas tienen filas duplicadas por distrito → conservar la última
    df = df.drop_duplicates(subset=["ubigeo", "fecha"], keep="last")
    if "log_poblacion" not in df.columns:
        df["log_poblacion"] = np.log1p(df["poblacion"].clip(lower=0))
    return df.reset_index(drop=True)


def resolver_semana_ref(df: pd.DataFrame) -> pd.Timestamp:
    """
    Última fecha con casos reales a nivel nacional. Las semanas posteriores
    en la AMT son relleno (ceros) para sostener los lags, no observaciones.
    """
    nac = df.groupby("fecha")["casos"].sum()
    con_senal = nac[nac > 0]
    return con_senal.index.max()


def semanas_historia(df: pd.DataFrame, ref: pd.Timestamp, n: int = N_HIST):
    """Lista ordenada de las últimas `n` fechas <= ref presentes en la AMT."""
    fechas = np.sort(df.loc[df["fecha"] <= ref, "fecha"].unique())
    return list(fechas[-n:])


def semanas_rango(df: pd.DataFrame, ref: pd.Timestamp, desde=None):
    """Todas las fechas en [desde, ref] presentes en la AMT (orden temporal)."""
    fechas = np.sort(df.loc[df["fecha"] <= ref, "fecha"].unique())
    if desde is not None:
        fechas = fechas[fechas >= np.datetime64(pd.Timestamp(desde))]
    return list(fechas)


def etiqueta_se(fecha) -> str:
    """Etiqueta 'SExx/AAAA' a partir de la semana epidemiológica ISO."""
    ts = pd.Timestamp(fecha)
    iso = ts.isocalendar()
    return f"SE{int(iso.week):02d}/{int(iso.year)}"


# ── Riesgo: mismos umbrales que el frontend (riskColor) ──────────────────────
UMBRALES_RIESGO = {"sin": 0, "bajo": 5, "moderado": 20, "alto": 100}


def banda_poisson(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IC95% aproximado tipo Poisson para modelos de punto (sin posterior)."""
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    sd = np.sqrt(p)
    lo = np.clip(p - 1.96 * sd, 0, None)
    hi = p + 1.96 * sd
    return lo, hi


def r(x, dec: int = 1):
    """Redondeo seguro a JSON (evita -0.0 y NaN)."""
    v = float(np.nan_to_num(x, nan=0.0))
    v = round(v, dec)
    return 0.0 if v == 0 else v
