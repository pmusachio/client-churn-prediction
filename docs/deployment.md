# Entrega e consumo

O projeto disponibiliza score de churn por API para integrar CRM, campanhas de retencao ou rotinas batch.

## Canais

- **FastAPI:** endpoint `/predict` para clientes em JSON.
- **Cloud web service:** `Procfile` com comando de inicializacao compativel com plataformas que usam processos web.
- **Batch:** comando `predict` para gerar CSV com score e predicao.

## API local

```bash
python -m pip install -r requirements.txt -r requirements-api.txt
PYTHONPATH=src python -m client_churn_prediction.cli train
PYTHONPATH=src uvicorn client_churn_prediction.api:app --reload
```

Teste em outro terminal:

```bash
PYTHONPATH=src python scripts/sample_api_request.py
```

## Cloud

Use o `Procfile` como referencia do processo web e configure o build para instalar `requirements.txt` e `requirements-api.txt`. Antes de expor o endpoint, garanta que `models/model.joblib` exista no ambiente ou seja gerado em uma etapa de build/job.
