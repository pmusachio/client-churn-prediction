# Dados

Fonte Kaggle: [Churn Dataset](https://www.kaggle.com/datasets/mervetorkan/churndataset).

Arquivos esperados nesta pasta:

- `churn.csv`

- O arquivo principal deve ficar em `data/raw/churn.csv`.

## Download via Kaggle API

```bash
mkdir -p data/raw
kaggle datasets download -d mervetorkan/churndataset --unzip -p data/raw
find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
```

Ajuste de nomes esperado pelo projeto:

```bash
mv data/raw/Churn_Modelling.csv data/raw/churn.csv 2>/dev/null || true
```

Mantenha arquivos grandes fora do Git quando necessario e baixe-os novamente no Colab ou no ambiente local.
