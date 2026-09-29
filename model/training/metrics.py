import os

import numpy as np
import pandas as pd

RESULT_COLUMNS = [
    "execucao",
    "timestamp",
    "vacina",
    "grupo",
    "fator",
    "valor",
    "seed",
    "window",
    "val_months",
    "test_months",
    "noise_level",
    "missing_ratio",
    "n_layers",
    "units",
    "dropout",
    "recurrent_dropout",
    "dense_units",
    "learning_rate",
    "epochs",
    "batch_size",
    "seasonal_m",
    "n_series",
    "n_train",
    "best_epoch",
    "val_rmse_escalado",
    "n_test",
    "mae",
    "rmse",
    "mape",
    "mape_n",
    "mase",
    "mase_n",
    "mase_m",
]


def naive_scale(in_sample, m):
    """Erro médio absoluto do preditor ingênuo sazonal y_t = y_{t-m} dentro da amostra de treino."""
    if len(in_sample) <= m:
        return None
    scale = float(np.mean(np.abs(in_sample[m:] - in_sample[:-m])))
    return scale if scale > 0 else None


def forecast_metrics(per_series, seasonal_m):
    """
    MAE, RMSE, MAPE e MASE agregados sobre todos os pontos de teste de todas as séries.

    per_series: lista de (real_teste, previsto_teste, treino_in_sample), todos em doses.

    - MAPE ignora meses com 0 doses reais (divisão por zero); `mape_n` diz quantos entraram.
    - MASE usa o `seasonal_m` pedido quando o treino tem mais de m meses; senão cai para
      m=1 (ingênuo simples). `mase_m` registra o m efetivamente usado. Séries cujo
      ingênuo tem erro zero no treino (constantes) ficam de fora.
    """
    real = np.concatenate([r for r, _, _ in per_series])
    predicted = np.concatenate([p for _, p, _ in per_series])
    errors = predicted - real

    nonzero = real > 0
    mape = float(np.mean(np.abs(errors[nonzero]) / real[nonzero]) * 100) if nonzero.any() else np.nan

    scaled_errors = []
    m_used = set()
    for series_real, series_pred, in_sample in per_series:
        m = seasonal_m if len(in_sample) > seasonal_m else 1
        scale = naive_scale(in_sample, m)
        if scale is None:
            continue
        m_used.add(m)
        scaled_errors.extend(np.abs(series_pred - series_real) / scale)

    return {
        "n_test": int(errors.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": mape,
        "mape_n": int(nonzero.sum()),
        "mase": float(np.mean(scaled_errors)) if scaled_errors else np.nan,
        "mase_n": len(scaled_errors),
        "mase_m": "/".join(str(m) for m in sorted(m_used)),
    }


def append_results(rows, path):
    """Acrescenta linhas ao registro acumulativo de resultados (nunca sobrescreve)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
