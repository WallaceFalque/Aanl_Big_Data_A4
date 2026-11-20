# Análise de Dados - Restaurantes Zomato (Bangalore)

Projeto de análise de dados com Python seguindo o processo OSEMN (Obter, Limpar, Explorar, Modelar, Interpretar), com clusterização, classificação e regressão.

## Dataset

https://www.kaggle.com/datasets/rajeshrampure/zomato-dataset

O arquivo original (`zomato.csv`, 574 MB) não está no repositório por exceder o limite do GitHub. Baixe-o do Kaggle e coloque em `dados/zomato.csv`. A base já tratada está em `dados/zomato_limpo.csv`.

## Execução

```
pip install -r requirements.txt
python main.py
```

A cada execução os gráficos são gerados com matplotlib, um por arquivo, na pasta `imagens/` (as imagens da execução anterior são apagadas). Os relatórios ficam em `relatorios/`.
