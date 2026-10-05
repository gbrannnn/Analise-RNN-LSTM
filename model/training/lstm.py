import os
import json
from dataclasses import dataclass, asdict, fields
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, YearLocator
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import Dense, InputLayer, LSTM
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed

from series import (
    build_supervised_dataset,
    find_vaccine_dirs,
    load_vaccine_series,
    perturb_series,
    scale,
    train_end_index,
    unscale,
)
from metrics import append_results, forecast_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPARED_DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "prepared_data", "pni", "SP")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "resultados.csv")

# Quantas séries plotar por vacina (as de maior volume de doses).
PLOT_TOP_N = 3

# Sem isso, duas rodadas com a mesma seed ainda podem divergir por ordem de operações
# não determinística, e a variação entre seeds deixaria de medir só a inicialização.
tf.config.experimental.enable_op_determinism()


@dataclass(frozen=True)
class ExperimentConfig:
    # Parâmetros de entrada.
    # Séries de 2024-01 a 2026-08 (32 meses): janela de 12 cobre a sazonalidade
    # anual e deixa treino prevendo 2025, validação 2026-01..04 e teste 2026-05..08.
    # Séries com menos de window + val + test + 1 meses (ex.: pastas antigas de 2020)
    # ficam de fora do treino.
    window: int = 12
    val_months: int = 4
    test_months: int = 4
    noise_level: float = 0.0
    missing_ratio: float = 0.0
    # Hiperparâmetros estruturais
    n_layers: int = 1
    units: int = 64
    dropout: float = 0.0
    recurrent_dropout: float = 0.0
    dense_units: int = 32
    learning_rate: float = 1e-3
    # Treino e avaliação
    epochs: int = 80
    batch_size: int = 32
    seasonal_m: int = 12
    seed: int = 42

    @classmethod
    def from_dict(cls, data):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def build_model(config):
    layers = [InputLayer(shape=(config.window, 1))]
    for i in range(config.n_layers):
        layers.append(
            LSTM(
                config.units,
                dropout=config.dropout,
                recurrent_dropout=config.recurrent_dropout,
                # Camadas LSTM empilhadas precisam receber a sequência inteira da anterior.
                return_sequences=i < config.n_layers - 1,
            )
        )
    layers += [
        Dense(config.dense_units, activation="relu"),
        Dense(1, activation="linear"),
    ]

    model = Sequential(layers)
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=config.learning_rate),
        metrics=[RootMeanSquaredError()],
    )
    return model


class RestoreBestWeights(Callback):
    """Guarda em memória os pesos da época de menor val_loss e os restaura ao fim do treino."""

    def on_train_begin(self, logs=None):
        self.best_loss = np.inf
        self.best_epoch = None
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs["val_loss"]
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def predict_series(model, series, scaler, window):
    """
    Previsão de 1 mês à frente para todos os meses previsíveis de uma série.

    Cada previsão usa apenas os `window` meses anteriores da própria série (não
    realimenta a saída do modelo), então o erro não se acumula.
    """
    mean, std = scaler
    scaled = scale(series.to_numpy(), mean, std)
    windows = np.asarray(
        [scaled[t - window:t] for t in range(window, len(scaled))], dtype="float32"
    ).reshape(-1, window, 1)

    predicted_scaled = model.predict(windows, verbose=0).reshape(-1)
    return series.index[window:], unscale(predicted_scaled, mean, std)


def evaluate_forecast(model, clean_series, model_series, scalers, config):
    """
    Métricas em doses sobre os meses de teste de todas as séries.

    As janelas de entrada vêm de `model_series` (o que o modelo enxerga, possivelmente
    perturbado); o alvo e o erro do ingênuo do MASE vêm de `clean_series` (a verdade).
    """
    codigos = [c for c in scalers if c in clean_series]
    windows, owners = [], []

    for codigo in codigos:
        mean, std = scalers[codigo]
        scaled = scale(model_series[codigo].to_numpy(), mean, std)
        n = len(scaled)
        for t in range(n - config.test_months, n):
            windows.append(scaled[t - config.window:t])
            owners.append(codigo)

    predicted_scaled = model.predict(
        np.asarray(windows, dtype="float32").reshape(-1, config.window, 1), verbose=0
    ).reshape(-1)

    per_series = []
    owners = np.asarray(owners)
    for codigo in codigos:
        mean, std = scalers[codigo]
        predicted = unscale(predicted_scaled[owners == codigo], mean, std)
        clean = clean_series[codigo].to_numpy()
        n = len(clean)
        in_sample = clean[:train_end_index(n, config.val_months, config.test_months)]
        per_series.append((clean[n - config.test_months:], predicted, in_sample))

    return forecast_metrics(per_series, config.seasonal_m)


def plot_predictions(model, clean_series, model_series, scalers, config, output_path, titulo):
    plotaveis = [c for c in scalers if c in clean_series]
    top = sorted(plotaveis, key=lambda c: clean_series[c].sum(), reverse=True)[:PLOT_TOP_N]
    if not top:
        return None

    fig, axes = plt.subplots(len(top), 1, figsize=(11, 3.2 * len(top)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, codigo in zip(axes, top):
        real = clean_series[codigo]
        dates, predicted = predict_series(model, model_series[codigo], scalers[codigo], config.window)

        ax.plot(real.index, real.to_numpy(), label="real", color="#1f77b4", marker="o", ms=3)
        ax.plot(dates, predicted, label="previsto", color="#d62728", ls="--", marker="x", ms=4)
        # Faixa dos meses de teste, que o modelo não viu durante o treino.
        ax.axvspan(real.index[-config.test_months], real.index[-1], color="#cccccc", alpha=0.35)

        ax.set_title(codigo, fontsize=10)
        ax.set_ylabel("doses")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(MonthLocator())

    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def run_experiment(nome_vacina, clean_series, config, save_dir=None, verbose=0):
    """
    Treina e avalia um modelo para uma vacina com uma configuração.

    Retorna a linha de resultado (config + métricas) ou None se não houver dados
    suficientes. Com `save_dir`, salva modelo, config e gráfico de previsões.
    """
    set_random_seed(config.seed)

    model_series = perturb_series(
        clean_series,
        config.noise_level,
        config.missing_ratio,
        config.val_months,
        config.test_months,
        config.seed,
    )
    datasets, scalers, skipped, _ = build_supervised_dataset(
        model_series, config.window, config.val_months, config.test_months
    )
    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]
    X_test, _ = datasets["test"]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        min_months = config.window + config.val_months + config.test_months + 1
        print(f"  dados insuficientes ({len(skipped)} séries com menos de {min_months} meses), pulando.")
        return None

    model = build_model(config)
    best = RestoreBestWeights()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        verbose=verbose,
        callbacks=[best],
    )

    _, val_rmse = model.evaluate(X_val, y_val, verbose=0)

    row = {
        "vacina": nome_vacina,
        **asdict(config),
        "n_series": len(scalers),
        "n_train": len(X_train),
        "best_epoch": best.best_epoch,
        "val_rmse_escalado": float(val_rmse),
        **evaluate_forecast(model, clean_series, model_series, scalers, config),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, "modelo.keras"))
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
        plot_predictions(
            model,
            clean_series,
            model_series,
            scalers,
            config,
            os.path.join(save_dir, "previsoes.png"),
            f"Doses aplicadas - real x previsto ({nome_vacina})",
        )

    return row


def load_saved_model(save_dir):
    with open(os.path.join(save_dir, "config.json"), encoding="utf-8") as f:
        config = ExperimentConfig.from_dict(json.load(f))
    return load_model(os.path.join(save_dir, "modelo.keras")), config


def baseline_dir(nome_vacina):
    return os.path.join(ARTIFACTS_DIR, nome_vacina, "baseline")


if __name__ == "__main__":
    # Treino único do baseline por vacina. Para várias seeds e a análise de
    # sensibilidade, use experiments.py.
    config = ExperimentConfig()
    execucao = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []

    for vaccine_dir in find_vaccine_dirs(PREPARED_DATA_DIR):
        nome_vacina = os.path.basename(vaccine_dir)
        print(f"\n=== vacina {nome_vacina} ===")
        clean_series = load_vaccine_series(vaccine_dir)

        row = run_experiment(nome_vacina, clean_series, config, save_dir=baseline_dir(nome_vacina), verbose=2)
        if row is None:
            continue

        row.update(
            execucao=execucao,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            grupo="baseline",
            fator="baseline",
            valor=np.nan,
        )
        print(
            f"MAE={row['mae']:.2f} RMSE={row['rmse']:.2f} MAPE={row['mape']:.2f}% "
            f"MASE={row['mase']:.3f} (m={row['mase_m']})"
        )
        rows.append(row)

    append_results(rows, RESULTS_PATH)
    if rows:
        print(f"\nresultados acrescentados em {RESULTS_PATH}")
        print(pd.DataFrame(rows)[["vacina", "n_test", "mae", "rmse", "mape", "mase", "mase_m"]].to_string(index=False))
