import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Práctica 4.2 - Regresión Lineal Múltiple", layout="wide")

st.title("Práctica de Laboratorio N° 4.2: Regresión Lineal y Correlación")
st.markdown(
    "**Asignatura:** Análisis y Exploración de Datos (`INC-2505`) | "
    "**Caso de Estudio:** Predicción del Consumo Per Cápita de Agua Potable (CPC)"
)

# Sidebar para parámetros interactivos (por defecto 0.42 como pide la guía)
st.sidebar.header("Configuración de Simulación")
seed_val = st.sidebar.number_input("Semilla determinista (DuckDB setseed)", min_value=0.01, max_value=0.99, value=0.42, step=0.01)
n_registros = st.sidebar.slider("Número de residencias simuladas", min_value=50, max_value=500, value=100, step=10)

# ==========================================
# PASO 1: SIMULACIÓN DETERMINISTA EN DUCKDB
# ==========================================
@st.cache_data
def generar_datos(seed: float, n: int) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute(f"SELECT setseed({seed});")
    con.execute(f"""
    CREATE OR REPLACE TABLE consumo_agua AS
    WITH base AS (
        SELECT
            row_number() OVER () AS residencia_id,
            20.0 + random() * 15.0 AS temperatura_max,
            30.0 + random() * 60.0 AS humedad_max,
            floor(2.0 + random() * 5.0)::DOUBLE AS personas_familia,
            0.5 + random() * 3.0 AS consumo_bidon_pc,
            random() AS u1,
            random() AS u2
        FROM range({n})
    ),
    con_error AS (
        SELECT
            residencia_id,
            round(temperatura_max, 2) AS temperatura_max,
            round(humedad_max, 2) AS humedad_max,
            personas_familia,
            round(consumo_bidon_pc, 2) AS consumo_bidon_pc,
            (sqrt(-2.0 * ln(greatest(u1, 1e-12))) * cos(2.0 * pi() * u2)) * 3.5 AS ruido
        FROM base
    )
    SELECT
        residencia_id,
        temperatura_max,
        humedad_max,
        personas_familia,
        consumo_bidon_pc,
        round(
            54.49
            + (2.46 * temperatura_max)
            - (0.79 * humedad_max)
            - (5.18 * personas_familia)
            + (14.00 * consumo_bidon_pc)
            + ruido,
            2
        ) AS cpc_consumo
    FROM con_error;
    """)
    return con.execute("SELECT * FROM consumo_agua").df()

df_agua = generar_datos(seed_val, n_registros)

st.subheader("Paso 1: Datos Simulados desde DuckDB")
st.dataframe(df_agua.head(10), use_container_width=True)

# ==========================================
# PASO 2: CORRELACIÓN DE PEARSON
# ==========================================
features = ["temperatura_max", "humedad_max", "personas_familia", "consumo_bidon_pc"]
target = "cpc_consumo"
matriz_correlacion = df_agua[features + [target]].corr(method="pearson")

st.subheader("Paso 2: Matriz de Correlación de Pearson (r)")
st.dataframe(matriz_correlacion.round(4), use_container_width=True)

# ==========================================
# PASO 3, 4 Y 5: MODELO OLS, SUPUESTOS Y MÉTRICAS
# ==========================================
X = df_agua[features]
Y = df_agua[target]
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()

residuos = modelo_ols.resid
y_pred = modelo_ols.fittedvalues

shapiro_stat, shapiro_p = stats.shapiro(residuos)
dw_stat = sm.stats.stattools.durbin_watson(residuos)

modelo_sk = LinearRegression().fit(X, Y)
y_pred_sk = modelo_sk.predict(X)
r2_pct = r2_score(Y, y_pred_sk) * 100
rmse_val = root_mean_squared_error(Y, y_pred_sk)

st.subheader("Paso 3 y 5: Métricas de Rendimiento y Ecuación Ajustada")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Coeficiente R²", f"{r2_pct:.2f}%")
col2.metric("RMSE", f"{rmse_val:.4f} lt/hab-día")
col3.metric("Shapiro-Wilk (p-valor)", f"{shapiro_p:.4f}")
col4.metric("Durbin-Watson", f"{dw_stat:.4f}")

st.info(
    f"**Ecuación del Modelo Ajustado:**  \n"
    f"`CPC = {modelo_ols.params['const']:.2f} + ({modelo_ols.params['temperatura_max']:.2f} * Temp) + "
    f"({modelo_ols.params['humedad_max']:.2f} * Humedad) + ({modelo_ols.params['personas_familia']:.2f} * Personas) + "
    f"({modelo_ols.params['consumo_bidon_pc']:.2f} * Bidón)`"
)

with st.expander("Ver Resumen Completo OLS (statsmodels)"):
    st.text(modelo_ols.summary().as_text())

# ==========================================
# PANEL DE 5 GRÁFICAS DIAGNÓSTICAS
# ==========================================
st.subheader("Panel de Diagnóstico Visual (Subtema 4.2)")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)", fontsize=16, fontweight="bold")

sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title("1. Matriz de Correlación de Pearson (r)")

sns.regplot(
    data=df_agua,
    x="humedad_max",
    y="cpc_consumo",
    color="b",
    ax=axes[0, 1],
    line_kws={"color": "red", "linewidth": 2},
)
axes[0, 1].set_title("2. Tendencia: Humedad Máx vs Consumo")
axes[0, 1].set_xlabel("Humedad Máxima (%)")
axes[0, 1].set_ylabel("CPC Consumo (lt/hab-día)")

axes[0, 2].scatter(y_pred, Y, alpha=0.7, color="teal")
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], "r--", lw=2)
axes[0, 2].set_title(f"3. Val. Reales vs Predichos (R² = {r2_pct:.1f}%)")
axes[0, 2].set_xlabel("CPC Predicho (ŷ)")
axes[0, 2].set_ylabel("CPC Real (y)")

axes[1, 0].scatter(y_pred, residuos, color="purple", alpha=0.6)
axes[1, 0].axhline(y=0, color="r", linestyle="--", lw=2)
axes[1, 0].set_title("4. Residuos vs Ajustados (Homocedasticidad)")
axes[1, 0].set_xlabel("Valores Predichos (ŷ)")
axes[1, 0].set_ylabel("Residuos (e = y - ŷ)")

sm.qqplot(residuos, line="45", fit=True, ax=axes[1, 1])
axes[1, 1].set_title("5. Gráfico Q-Q Normal (Residuos)")

axes[1, 2].axis("off")
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.38, wspace=0.3)

st.pyplot(fig)