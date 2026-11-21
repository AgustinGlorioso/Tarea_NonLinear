import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


def cargar_ventas():
    datos = []
    for s in ["A", "B", "C", "D", "E"]:
        datos.append(pd.read_excel(f"../Datos/ventas_SKU_P{s}_hasta_mayo.xlsx"))
    df = pd.concat(datos)
    df["SKU"] = df["SKU"].str.replace("Producto", "")
    return df.pivot(index="Fecha", columns="SKU", values="VentasMensuales")


def naive(train, h):
    return [train.iloc[-1]] * h


def seasonal_naive(train, h):
    return [train.iloc[-12 + i] for i in range(h)]


def media(train, h):
    return [train.mean()] * h


def media_movil(train, h, k):
    return [train.iloc[-k:].mean()] * h


def ses(train, h):
    modelo = SimpleExpSmoothing(train.values, initialization_method="estimated").fit()
    return list(modelo.forecast(h))


def holt(train, h):
    modelo = Holt(train.values, initialization_method="estimated").fit()
    return list(modelo.forecast(h))


def holt_amortiguado(train, h):
    modelo = Holt(train.values, damped_trend=True, initialization_method="estimated").fit()
    return list(modelo.forecast(h))


def holt_winters(train, h):
    modelo = ExponentialSmoothing(train.values, seasonal="add", seasonal_periods=12,
                                  initialization_method="estimated").fit()
    return list(modelo.forecast(h))


def holt_winters_amortiguado(train, h):
    modelo = ExponentialSmoothing(train.values, trend="add", damped_trend=True, seasonal="add",
                                  seasonal_periods=12, initialization_method="estimated").fit()
    return list(modelo.forecast(h))

def holt_parametrizado(train, h, alpha, beta):
    modelo = Holt(
        train.values,
        initialization_method="estimated"
    ).fit(
        smoothing_level=alpha,
        smoothing_trend=beta,
        optimized=False
    )

    return list(modelo.forecast(h))


def holt_amortiguado_parametrizado(train, h, alpha, beta, phi):
    modelo = Holt(
        train.values,
        damped_trend=True,
        initialization_method="estimated"
    ).fit(
        smoothing_level=alpha,
        smoothing_trend=beta,
        damping_trend=phi,
        optimized=False
    )

    return list(modelo.forecast(h))

def walk_forward(ventas, skus, modelos, cortes, horizonte=3):
    filas = []
    for s in skus:
        for corte in cortes:
            train = ventas.loc[:corte, s]
            real = ventas.loc[corte:, s].iloc[1:horizonte + 1]
            escala = train.diff().abs().mean()
            for nombre, modelo in modelos.items():
                pronostico = modelo(train, horizonte)
                for i in range(horizonte):
                    filas.append({
                        "SKU": s,
                        "Modelo": nombre,
                        "Corte": corte,
                        "Horizonte": i + 1,
                        "Fecha": real.index[i],
                        "Real": real.iloc[i],
                        "Pronostico": pronostico[i],
                        "Escala": escala,
                    })
    resultados = pd.DataFrame(filas)
    resultados["Error"] = resultados["Pronostico"] - resultados["Real"]
    resultados["ErrorAbs"] = resultados["Error"].abs()
    return resultados


def metricas(g):
    return pd.Series({
        "MAE": g["ErrorAbs"].mean(),
        "MASE": (g["ErrorAbs"] / g["Escala"]).mean(),
        "Sesgo": g["Error"].mean(),
    })
