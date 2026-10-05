

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from shared import features as ft

PASTA_MODELO = Path(os.getenv("CAMINHO_MODELO", "/model"))
CSV_DADOS = Path(os.getenv("CAMINHO_CSV", "/data/btcusdt_1d.csv"))


# ----------------------------------------------------------------- contratos

class Candle(BaseModel):
    date: str = Field(description="data do candle, formato YYYY-MM-DD")
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("open", "high", "low", "close")
    @classmethod
    def precos_positivos(cls, valor: float) -> float:
        if valor <= 0:
            raise ValueError("preco precisa ser maior que zero")
        return valor


class PedidoPredicao(BaseModel):
    candles: list[Candle] | None = Field(
        default=None,
        description="historico diario em ordem cronologica. Vazio = usa o CSV do container.",
    )


class RespostaPredicao(BaseModel):
    par: str
    data_referencia: str
    fechamento_referencia: float
    data_prevista: str
    fechamento_previsto: float
    variacao_prevista_pct: float
    direcao: str
    versao_modelo: str
    origem_dados: str


class RespostaSaude(BaseModel):
    status: str
    modelo_carregado: bool
    versao_modelo: str | None
    treinado_em: str | None


# -------------------------------------------------------------------- modelo

class Preditor:
    """Guarda o pipeline carregado do disco e faz a predicao."""

    def __init__(self) -> None:
        self.pipeline = None
        self.colunas: list[str] = []
        self.metadados: dict = {}
        self.erro: str | None = None

    def carregar(self) -> None:
        arquivo = PASTA_MODELO / "model.joblib"
        if not arquivo.exists():
            self.pipeline = None
            self.erro = f"{arquivo} nao encontrado, rode o container de treino primeiro"
            return
        artefato = joblib.load(arquivo)
        self.pipeline = artefato["pipeline"]
        self.colunas = artefato["colunas_features"]

        metadata = PASTA_MODELO / "metadata.json"
        self.metadados = (
            json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
        )
        self.erro = None

    @property
    def pronto(self) -> bool:
        return self.pipeline is not None

    def _historico_do_disco(self) -> tuple[pd.DataFrame, str]:
        # o CSV montado e a fonte principal; a janela salva junto do modelo e o plano B
        for caminho in (CSV_DADOS, PASTA_MODELO / "ultima_janela.csv"):
            if caminho.exists():
                df = ft.carregar_csv(caminho)
                return ft.remover_candle_incompleto(df), f"csv:{caminho}"
        raise RuntimeError("nenhuma fonte de dados disponivel no container")

    def prever(self, candles: list[dict] | None) -> dict:
        if candles:
            df = ft.carregar_csv_de_registros(candles)
            origem = f"payload:{len(candles)}_candles"
        else:
            df, origem = self._historico_do_disco()

        if len(df) < ft.LINHAS_MINIMAS:
            raise ValueError(
                f"preciso de pelo menos {ft.LINHAS_MINIMAS} candles diarios, recebi {len(df)}"
            )

        X, data_ref, close_ref = ft.features_do_ultimo_dia(df)
        retorno = float(self.pipeline.predict(X[self.colunas])[0])
        preco = close_ref * float(np.exp(retorno))
        variacao = (preco / close_ref - 1.0) * 100

        if variacao > 0.05:
            direcao = "alta"
        elif variacao < -0.05:
            direcao = "baixa"
        else:
            direcao = "estavel"

        return {
            "par": self.metadados.get("par", "BTCUSDT"),
            "data_referencia": str(pd.Timestamp(data_ref).date()),
            "fechamento_referencia": round(close_ref, 2),
            "data_prevista": str((pd.Timestamp(data_ref) + timedelta(days=1)).date()),
            "fechamento_previsto": round(preco, 2),
            "variacao_prevista_pct": round(variacao, 4),
            "direcao": direcao,
            "versao_modelo": self.metadados.get("versao_modelo", "desconhecida"),
            "origem_dados": origem,
        }


preditor = Preditor()


# --------------------------------------------------------------------- rotas

app = FastAPI(
    title="API de predicao BTC/USDT",
    description="Predicao experimental, nao serve como recomendacao de investimento.",
    version="1.0.0",
)


@app.on_event("startup")
def iniciar() -> None:
    preditor.carregar()
    print(
        f"modelo carregado, versao {preditor.metadados.get('versao_modelo')}"
        if preditor.pronto
        else f"subi sem modelo: {preditor.erro}"
    )


@app.get("/", include_in_schema=False)
def raiz() -> dict:
    return {"servico": "api-predicao-btc", "rotas": ["/health", "/model/info", "/predict", "/docs"]}


@app.get("/health", response_model=RespostaSaude)
def health() -> RespostaSaude:
    # responde 200 mesmo sem modelo, com modelo_carregado=false, para dar pra
    # diferenciar "o processo morreu" de "o artefato nao chegou no volume"
    return RespostaSaude(
        status="ok" if preditor.pronto else "sem_modelo",
        modelo_carregado=preditor.pronto,
        versao_modelo=preditor.metadados.get("versao_modelo"),
        treinado_em=preditor.metadados.get("treinado_em"),
    )


@app.get("/model/info")
def info_modelo() -> JSONResponse:
    if not preditor.pronto:
        raise HTTPException(status_code=503, detail=preditor.erro)
    return JSONResponse(preditor.metadados)


@app.post("/model/reload", response_model=RespostaSaude)
def recarregar() -> RespostaSaude:
    # o modelo e lido so no startup, entao sem isso a API fica respondendo com
    # o artefato antigo depois de um retreino
    preditor.carregar()
    return health()


@app.post("/predict", response_model=RespostaPredicao)
def predict(pedido: PedidoPredicao) -> RespostaPredicao:
    if not preditor.pronto:
        raise HTTPException(status_code=503, detail=preditor.erro or "modelo nao carregado")

    candles = [c.model_dump() for c in pedido.candles] if pedido.candles else None
    try:
        return RespostaPredicao(**preditor.prever(candles))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
