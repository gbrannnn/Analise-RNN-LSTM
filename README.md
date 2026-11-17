# TCC — Previsão mensal de doses de imunobiológicos por município de SP com LSTM

Previsão, um mês à frente, das doses aplicadas de quatro imunobiológicos do PNI (códigos 9, 15, 33 e 67) em cada município de São Paulo, usando redes LSTM, dados abertos do Ministério da Saúde, análise de sensibilidade e explicabilidade (SHAP e LIME).

**Pergunta de pesquisa.** Um LSTM prevê as doses mensais por município melhor do que repetir o mesmo mês do ano anterior (ingênuo sazonal)? Critério de aceite: **MASE < 1**.

## Resultado em uma tabela

Média de 5 seeds, 4 meses de teste (2026-05 a 2026-08).

| Vacina | Séries | MASE sem exógenas | MASE com exógenas | Aceita (MASE < 1) |
|---|---|---|---|---|
| 9 | 361 | 0,641 | 0,640 | sim |
| 15 | 162 | 0,723 | 0,727 | sim |
| 67 | 232 | 0,763 | 0,758 | sim |
| 33 | 616 | 1,934 | 1,316 | **não** |

Em resumo: o LSTM supera o ingênuo sazonal em 3 de 4 vacinas. A 33, de volume alto e concentrado, não passa. A janela de entrada é o fator mais influente na sensibilidade, e as variáveis exógenas só ajudam na 33. Limites principais: séries curtas (32 meses, 12 amostras de treino por município) e uma única referência de comparação. Detalhes e ressalvas em [DOCUMENTACAO.md](DOCUMENTACAO.md).

## Estrutura

```
data/
  data_collect/     coleta na API do PNI (JSON bruto em data/raw_data)
  data_preparing/   série mensal por (vacina, município) + covariáveis exógenas
  prepared_data/    CSVs prontos (pni e pni_exogenas)
model/
  training/         series.py, lstm.py, metrics.py, experiments.py, explain.py
  artifacts*/       resultados de cada rodada (cada pasta tem resume.md)
apresentacao/       slides (build.py gera o .pptx) e figuras de apoio
```

## Documentação

| Arquivo | Conteúdo |
|---|---|
| [DOCUMENTACAO.md](DOCUMENTACAO.md) | O que foi feito e por quê: dados, modelo, métricas, resultados, limitações. |
| [ROTEIRO_APRESENTACAO.md](ROTEIRO_APRESENTACAO.md) | Roteiro slide a slide e perguntas prováveis da banca. |
| [ESTUDO_APRESENTACAO.md](ESTUDO_APRESENTACAO.md) | Guia de estudo: métricas com exemplos, justificativas e fraquezas. |
| `data/data_preparing/resume.md` | Decisões da preparação dos dados. |
| `model/training/resume.md` | Decisões do treino, métricas, sensibilidade e explicabilidade. |
| `model/artifacts*/resume.md` | Resultados e leitura de cada rodada. |

## Rodadas de resultados

| Pasta | Conteúdo |
|---|---|
| `artifacts_wind4`, `artifacts_01` | Rodadas antigas (janela 4, séries curtas); só histórico, **não comparáveis** com as atuais. |
| `artifacts_02`, `artifacts_baseline` | Rodada principal sem exógenas: baseline, sensibilidade e SHAP/LIME. |
| `artifacts_exogenas` | Teste de 1 seed com covariáveis. |
| `artifacts_exogenas_baseline` | Baseline com covariáveis, 5 seeds. |
| `artifacts` | Modelo com covariáveis: baseline, sensibilidade completa e SHAP/LIME das 4 vacinas (2026-11-07). |

## Como reproduzir

```bash
pip install -r requirements.txt
python data/data_collect/main.py            # coleta (demorada); caminho de saída fixo em worker.py
python data/data_preparing/main.py          # gera prepared_data/pni e pni_exogenas
cd model/training
python lstm.py                              # treino único do baseline
python experiments.py --apenas-baseline     # baseline com 5 seeds
python experiments.py                       # baseline + sensibilidade OFAT
python explain.py                           # SHAP e LIME
# modelo com covariáveis: acrescente --dados exogenas em cada comando
```

O código sempre escreve em `model/artifacts/`; as demais pastas `artifacts_*` são cópias renomeadas de rodadas anteriores.

## Slides

```bash
python apresentacao/build.py
```
