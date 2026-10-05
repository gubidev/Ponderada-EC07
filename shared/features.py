

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LAGS_RETORNO = (1, 2, 3, 5, 10)
JANELAS_MEDIA = (7, 21)
JANELA_VOLATILIDADE = 7
JANELA_VOLUME = 7

# 21 dias para a media movel mais longa + 1 para o retorno do dia
LINHAS_MINIMAS = 22

COLUNAS_FEATURES: list[str] = (
    [f"ret_lag_{lag}" for lag in LAGS_RETORNO]
    + [f"desvio_sma_{janela}" for janela in JANELAS_MEDIA]
    + ["volatilidade_7", "desvio_volume_7", "amplitude_relativa"]
)

COLUNAS_CSV = ["date", "open", "high", "low", "close", "volume"]


def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Valida colunas, converte tipos, remove duplicatas e ordena por data."""
    faltando = [coluna for coluna in COLUNAS_CSV if coluna not in df.columns]
    if faltando:
        raise ValueError(f"dados sem as colunas obrigatorias: {faltando}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    for coluna in ("open", "high", "low", "close", "volume"):
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = df.dropna(subset=["close"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def carregar_csv(caminho: str | Path) -> pd.DataFrame:
    """Le o CSV de candles (a base de dados do projeto)."""
    return _normalizar(pd.read_csv(caminho))


def carregar_csv_de_registros(registros: list[dict]) -> pd.DataFrame:
    """Mesmo tratamento do CSV, mas para candles que chegam no corpo do request."""
    return _normalizar(pd.DataFrame(registros))


def remover_candle_incompleto(df: pd.DataFrame, agora: datetime | None = None) -> pd.DataFrame:
    """
    Descarta o candle do dia corrente.

    A Binance devolve o dia de hoje ainda aberto, com o fechamento igual ao
    ultimo preco negociado. Treinar com esse candle vaza informacao parcial e,
    na inferencia, usar ele como referencia faz a predicao mudar a cada request.
    """
    if df.empty:
        return df
    agora = agora or datetime.now(timezone.utc)
    hoje = pd.Timestamp(agora.date(), tz="UTC")
    return df[df["date"] < hoje].reset_index(drop=True)


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monta as features de cada dia usando somente informacao daquele dia e dos
    anteriores. Nada de olhar para o futuro.
    """
    if len(df) < LINHAS_MINIMAS:
        raise ValueError(
            f"preciso de pelo menos {LINHAS_MINIMAS} candles para montar as features, recebi {len(df)}"
        )

    tabela = df.copy()
    tabela["retorno"] = np.log(tabela["close"]).diff()

    for lag in LAGS_RETORNO:
        tabela[f"ret_lag_{lag}"] = np.log(tabela["close"] / tabela["close"].shift(lag))

    for janela in JANELAS_MEDIA:
        media = tabela["close"].rolling(janela).mean()
        tabela[f"desvio_sma_{janela}"] = tabela["close"] / media - 1.0

    tabela["volatilidade_7"] = tabela["retorno"].rolling(JANELA_VOLATILIDADE).std()

    media_volume = tabela["volume"].rolling(JANELA_VOLUME).mean()
    tabela["desvio_volume_7"] = np.where(
        media_volume > 0, tabela["volume"] / media_volume - 1.0, 0.0
    )

    tabela["amplitude_relativa"] = (tabela["high"] - tabela["low"]) / tabela["close"]

    return tabela


def montar_dataset_supervisionado(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Devolve (X, y, referencia).

    y e o retorno em log do dia seguinte: log(close[t+1] / close[t]). Previ o
    retorno em vez do preco porque o preco do BTC nao e estacionario, vai de
    40k a 120k na base. Um modelo linear treinado direto no preco aprende so a
    repetir o ultimo valor e nao generaliza para faixas que nunca viu.

    referencia guarda date e close de cada linha, usado depois para reconstruir
    o preco previsto e calcular as metricas em dolar.
    """
    tabela = construir_features(df)
    tabela["alvo_retorno"] = np.log(tabela["close"].shift(-1) / tabela["close"])

    colunas = COLUNAS_FEATURES + ["alvo_retorno", "date", "close"]
    tabela = tabela[colunas].dropna().reset_index(drop=True)

    X = tabela[COLUNAS_FEATURES]
    y = tabela["alvo_retorno"]
    referencia = tabela[["date", "close"]]
    return X, y, referencia


def features_do_ultimo_dia(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    """
    Vetor de features do dia mais recente da serie, que e o usado para prever o
    fechamento do dia seguinte. Devolve tambem a data e o fechamento de
    referencia.
    """
    tabela = construir_features(df)
    validas = tabela[COLUNAS_FEATURES].dropna()
    if validas.empty:
        raise ValueError("nao consegui montar nenhuma linha completa de features")

    indice = validas.index[-1]
    X = validas.loc[[indice]]
    return X, tabela.loc[indice, "date"], float(tabela.loc[indice, "close"])
