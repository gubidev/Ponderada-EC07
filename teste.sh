#!/usr/bin/env bash


set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
falhas=0

checar() {
  local nome="$1" esperado="$2" recebido="$3"
  if [[ "$recebido" == "$esperado" ]]; then
    echo "  ok    $nome (HTTP $recebido)"
  else
    echo "  FALHA $nome (esperava $esperado, veio $recebido)"
    falhas=$((falhas + 1))
  fi
}

echo "1) healthcheck"
corpo=$(curl -s "$BASE/health")
codigo=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")
echo "     $corpo"
checar "/health responde" 200 "$codigo"

echo "2) metadados do modelo"
codigo=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/model/info")
checar "/model/info responde" 200 "$codigo"

echo "3) predicao usando o CSV montado no container"
corpo=$(curl -s -X POST "$BASE/predict" -H 'Content-Type: application/json' -d '{}')
codigo=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d '{}')
echo "     $corpo"
checar "/predict sem corpo" 200 "$codigo"

echo "4) predicao com candles vindos do cliente"
payload=$(python3 - <<'PY'
import csv, json
linhas = list(csv.DictReader(open("data/btcusdt_1d.csv")))[-30:]
candles = [
    {
        "date": l["date"],
        "open": float(l["open"]),
        "high": float(l["high"]),
        "low": float(l["low"]),
        "close": float(l["close"]),
        "volume": float(l["volume"]),
    }
    for l in linhas
]
print(json.dumps({"candles": candles}))
PY
)
codigo=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d "$payload")
checar "/predict com 30 candles" 200 "$codigo"

echo "5) historico curto demais deve ser rejeitado"
curto='{"candles":[{"date":"2026-10-01","open":1,"high":2,"low":1,"close":1.5,"volume":10}]}'
codigo=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d "$curto")
checar "/predict com 1 candle" 422 "$codigo"

echo "6) preco negativo deve ser rejeitado pelo schema"
invalido='{"candles":[{"date":"2026-10-01","open":1,"high":2,"low":1,"close":-5,"volume":10}]}'
codigo=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d "$invalido")
checar "/predict com close negativo" 422 "$codigo"

echo
if [[ $falhas -eq 0 ]]; then
  echo "todos os testes passaram"
else
  echo "$falhas teste(s) falharam"
fi
exit $falhas
