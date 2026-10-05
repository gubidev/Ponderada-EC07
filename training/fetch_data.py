

import argparse
import csv
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def baixar_klines(symbol: str, interval: str, limit: int) -> list[list]:
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url, timeout=30) as resposta:
        if resposta.status != 200:
            raise RuntimeError(f"Binance respondeu {resposta.status}")
        return json.loads(resposta.read().decode("utf-8"))


def salvar_csv(klines: list[list], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for k in klines:
            abertura_ms = int(k[0])
            data = datetime.fromtimestamp(abertura_ms / 1000, tz=timezone.utc)
            writer.writerow([
                data.strftime("%Y-%m-%d"),
                k[1], k[2], k[3], k[4], k[5],
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--limit", type=int, default=1000, help="maximo 1000 por request")
    parser.add_argument("--saida", default="data/btcusdt_1d.csv")
    args = parser.parse_args()

    klines = baixar_klines(args.symbol, args.interval, args.limit)
    destino = Path(args.saida)
    salvar_csv(klines, destino)
    print(f"{len(klines)} candles salvos em {destino}")
    print(f"periodo: {klines[0][0]} -> {klines[-1][0]} (epoch ms)")


if __name__ == "__main__":
    main()
