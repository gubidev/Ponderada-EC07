
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared import features as ft

VERSAO_MODELO = "1.0.0"
GRADE_ALPHA = [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]


def separar_cronologico(X, y, ref, dias_validacao: int, dias_teste: int) -> dict:
   
    fim_treino = len(X) - dias_validacao - dias_teste
    fim_validacao = len(X) - dias_teste
    if fim_treino <= ft.LINHAS_MINIMAS:
        raise ValueError("base pequena demais para separar treino, validacao e teste")

    fatias = {
        "treino": slice(0, fim_treino),
        "validacao": slice(fim_treino, fim_validacao),
        "teste": slice(fim_validacao, len(X)),
    }
    return {n: (X.iloc[f], y.iloc[f], ref.iloc[f]) for n, f in fatias.items()}


def treinar(X, y, alpha: float) -> Pipeline:
    modelo = Pipeline([("escala", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    modelo.fit(X, y)
    return modelo


def metricas(close_ref, retorno_real, retorno_previsto, com_direcao: bool = True) -> dict:
    """Erro medido em dolar, que e o numero que faz sentido olhar."""
    preco_real = close_ref * np.exp(retorno_real)
    erro = close_ref * np.exp(retorno_previsto) - preco_real

    return {
        "mae_usd": float(np.mean(np.abs(erro))),
        "rmse_usd": float(np.sqrt(np.mean(erro ** 2))),
        "mape_pct": float(np.mean(np.abs(erro / preco_real)) * 100),
        # a baseline preve variacao zero, nao tem direcao para acertar
        "acerto_direcao_pct": float(
            np.mean(np.sign(retorno_previsto) == np.sign(retorno_real)) * 100
        ) if com_direcao else None,
    }


def escolher_alpha(dados: dict) -> tuple[float, list[dict]]:
    X_tr, y_tr, _ = dados["treino"]
    X_val, y_val, ref_val = dados["validacao"]

    historico = []
    for alpha in GRADE_ALPHA:
        previsto = treinar(X_tr, y_tr, alpha).predict(X_val)
        m = metricas(ref_val["close"].to_numpy(), y_val.to_numpy(), previsto)
        historico.append({"alpha": alpha, **m})
        print(f"      alpha={alpha:>7} | MAE val = {m['mae_usd']:9.2f} USD"
              f" | direcao = {m['acerto_direcao_pct']:.1f}%")

    melhor = min(historico, key=lambda linha: linha["mae_usd"])
    return melhor["alpha"], historico


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/btcusdt_1d.csv")
    parser.add_argument("--saida", default="model")
    parser.add_argument("--dias-teste", type=int, default=150)
    parser.add_argument("--dias-validacao", type=int, default=150)
    parser.add_argument("--alpha", default="auto", help="'auto' escolhe pela validacao")
    args = parser.parse_args()

    print(f"[1/5] lendo {args.csv}")
    df = ft.carregar_csv(args.csv)
    bruto = len(df)
    df = ft.remover_candle_incompleto(df)
    print(f"      {bruto} linhas no CSV, {len(df)} depois de tirar o candle do dia corrente")
    print(f"      periodo: {df['date'].min().date()} -> {df['date'].max().date()}")

    print("[2/5] montando features e alvo")
    X, y, ref = ft.montar_dataset_supervisionado(df)
    print(f"      {len(X)} amostras x {len(ft.COLUNAS_FEATURES)} features")

    dados = separar_cronologico(X, y, ref, args.dias_validacao, args.dias_teste)
    for nome, (parte, _, ref_parte) in dados.items():
        print(f"      {nome:<10} {len(parte):>4} dias "
              f"({ref_parte['date'].min().date()} -> {ref_parte['date'].max().date()})")

    print("[3/5] escolhendo o alpha na validacao")
    if args.alpha == "auto":
        alpha, busca = escolher_alpha(dados)
        print(f"      alpha escolhido: {alpha}")
    else:
        alpha, busca = float(args.alpha), []
        print(f"      alpha fixado na linha de comando: {alpha}")

    X_final = pd.concat([dados["treino"][0], dados["validacao"][0]])
    y_final = pd.concat([dados["treino"][1], dados["validacao"][1]])
    ref_final = pd.concat([dados["treino"][2], dados["validacao"][2]])
    modelo = treinar(X_final, y_final, alpha)

    print(f"[4/5] avaliando no teste (treino final com {len(X_final)} dias)")
    X_teste, y_teste, ref_teste = dados["teste"]
    previsto = modelo.predict(X_teste)
    close_ref = ref_teste["close"].to_numpy()

    m_modelo = metricas(close_ref, y_teste.to_numpy(), previsto)
    m_baseline = metricas(close_ref, y_teste.to_numpy(), np.zeros_like(previsto), False)
    ganho = (m_baseline["mae_usd"] - m_modelo["mae_usd"]) / m_baseline["mae_usd"] * 100

    print(f"      modelo   -> MAE {m_modelo['mae_usd']:.2f} USD | RMSE {m_modelo['rmse_usd']:.2f}"
          f" | MAPE {m_modelo['mape_pct']:.3f}% | direcao {m_modelo['acerto_direcao_pct']:.1f}%")
    print(f"      baseline -> MAE {m_baseline['mae_usd']:.2f} USD | RMSE {m_baseline['rmse_usd']:.2f}"
          f" | MAPE {m_baseline['mape_pct']:.3f}%")
    print(f"      MAE do modelo em relacao a baseline: {ganho:+.2f}%")

    print(f"[5/5] salvando artefato em {args.saida}")
    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "pipeline": modelo,
        "colunas_features": ft.COLUNAS_FEATURES,
        "versao_modelo": VERSAO_MODELO,
        "alvo": "log_retorno_dia_seguinte",
    }, pasta / "model.joblib")

    metadados = {
        "versao_modelo": VERSAO_MODELO,
        "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algoritmo": f"StandardScaler + Ridge(alpha={alpha})",
        "alvo": "log(close[t+1] / close[t])",
        "par": "BTCUSDT",
        "intervalo": "1d",
        "fonte_dados": "klines diarios da API publica da Binance, CSV em data/btcusdt_1d.csv",
        "janelas": {
            nome: {
                "inicio": str(ref_parte["date"].min().date()),
                "fim": str(ref_parte["date"].max().date()),
                "dias": len(parte),
            }
            for nome, (parte, _, ref_parte) in dados.items()
        },
        "treino_final": {
            "inicio": str(ref_final["date"].min().date()),
            "fim": str(ref_final["date"].max().date()),
            "dias": len(X_final),
        },
        "features": ft.COLUNAS_FEATURES,
        "coeficientes": dict(zip(
            ft.COLUNAS_FEATURES,
            (float(c) for c in modelo.named_steps["ridge"].coef_),
        )),
        "busca_alpha_validacao": busca,
        "metricas_teste": m_modelo,
        "metricas_baseline_ultimo_preco": m_baseline,
        "ganho_mae_sobre_baseline_pct": round(ganho, 2),
        "versoes": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (pasta / "metadata.json").write_text(
        json.dumps(metadados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


    janela = df.tail(ft.LINHAS_MINIMAS + 5).copy()
    janela["date"] = janela["date"].dt.strftime("%Y-%m-%d")
    janela.to_csv(pasta / "ultima_janela.csv", index=False)

    print("      ok: model.joblib, metadata.json e ultima_janela.csv gravados")


if __name__ == "__main__":
    main()
