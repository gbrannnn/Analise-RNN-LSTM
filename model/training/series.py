import os

import numpy as np
import pandas as pd


def find_vaccine_dirs(prepared_dir):
    if not os.path.isdir(prepared_dir):
        raise FileNotFoundError(
            f"Diretório de dados preparados não encontrado: {prepared_dir}. "
            "Rode data/data_preparing/main.py antes do treino."
        )

    dirs = [
        os.path.join(prepared_dir, name)
        for name in sorted(os.listdir(prepared_dir))
        if os.path.isdir(os.path.join(prepared_dir, name))
    ]
    if not dirs:
        raise FileNotFoundError(f"Nenhuma pasta de vacina em {prepared_dir}")
    return dirs


def load_vaccine_series(vaccine_dir):
    """Carrega todas as séries (município -> doses por mês) de uma vacina."""
    series_by_municipio = {}

    for filename in sorted(os.listdir(vaccine_dir)):
        if not filename.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(vaccine_dir, filename), parse_dates=["mes"])
        series = df.set_index("mes")["doses_aplicadas"].sort_index().astype("float64")

        # A janela deslizante só representa meses consecutivos se o índice for contíguo.
        expected = pd.date_range(series.index.min(), series.index.max(), freq="MS")
        if not series.index.equals(expected):
            print(f"  aviso: {filename} tem meses faltando, pulando.")
            continue

        series_by_municipio[os.path.splitext(filename)[0]] = series

    return series_by_municipio


def train_end_index(n, val_months, test_months):
    return n - val_months - test_months


def impute(series):
    filled = series.interpolate(method="linear", limit_direction="both")
    # Série inteira mascarada não tem vizinho para interpolar.
    return filled.fillna(0.0)


def perturb_series(series_by_municipio, noise_level, missing_ratio, val_months, test_months, seed):
    """
    Degrada as séries para os experimentos de sensibilidade dos parâmetros de entrada.

    - noise_level: ruído gaussiano com desvio = noise_level * desvio da série no treino.
    - missing_ratio: fração dos meses mascarada e depois imputada por interpolação linear.

    O modelo treina e prevê a partir da série degradada, mas as métricas são
    sempre calculadas contra a série original.
    """
    if noise_level == 0 and missing_ratio == 0:
        return series_by_municipio

    rng = np.random.default_rng(seed)
    perturbed = {}

    for codigo_municipio, series in series_by_municipio.items():
        values = series.to_numpy(copy=True)
        n = len(values)

        if noise_level > 0:
            train_end = max(train_end_index(n, val_months, test_months), 1)
            sigma = noise_level * values[:train_end].std()
            values = np.clip(values + rng.normal(0.0, sigma, n), 0.0, None)

        if missing_ratio > 0:
            n_missing = int(round(missing_ratio * n))
            values[rng.choice(n, size=n_missing, replace=False)] = np.nan

        result = pd.Series(values, index=series.index)
        perturbed[codigo_municipio] = impute(result) if missing_ratio > 0 else result

    return perturbed


def fit_scaler(train_values):
    """
    Estatísticas de normalização de uma série, calculadas só com os meses de treino.

    log1p comprime a diferença de escala entre municípios grandes e pequenos e a
    padronização por série deixa todas na mesma faixa, o que permite treinar um
    único modelo com os municípios agrupados.
    """
    log_values = np.log1p(train_values)
    mean = float(log_values.mean())
    std = float(log_values.std())
    # Série constante no treino (ex.: só zeros) daria divisão por zero.
    if std < 1e-8:
        std = 1.0
    return mean, std


def scale(values, mean, std):
    return (np.log1p(values) - mean) / std


def unscale(scaled_values, mean, std):
    doses = np.expm1(scaled_values * std + mean)
    return np.clip(doses, 0.0, None)


def build_supervised_dataset(series_by_municipio, window, val_months, test_months):
    """
    Monta as janelas (X, y) de todas as séries de uma vacina em um único dataset.

    O split é temporal dentro de cada série: uma amostra vai para treino,
    validação ou teste conforme o mês que ela prevê. As janelas de entrada podem
    incluir meses de treino — o que nunca acontece é o alvo de um mês de teste
    influenciar o ajuste dos pesos ou as estatísticas de normalização.

    `meta[split]` guarda (município, mês alvo) de cada amostra, na mesma ordem de X.
    """
    splits = {name: {"X": [], "y": [], "meta": []} for name in ("train", "val", "test")}
    scalers = {}
    skipped = []

    min_months = window + val_months + test_months + 1

    for codigo_municipio, series in series_by_municipio.items():
        values = series.to_numpy()
        n = len(values)

        if n < min_months:
            skipped.append((codigo_municipio, n))
            continue

        train_end = train_end_index(n, val_months, test_months)
        mean, std = fit_scaler(values[:train_end])
        scaled = scale(values, mean, std)
        scalers[codigo_municipio] = (mean, std)

        for t in range(window, n):
            if t < train_end:
                split = "train"
            elif t < train_end + val_months:
                split = "val"
            else:
                split = "test"

            splits[split]["X"].append(scaled[t - window:t])
            splits[split]["y"].append(scaled[t])
            splits[split]["meta"].append((codigo_municipio, series.index[t]))

    datasets, meta = {}, {}
    for split, data in splits.items():
        meta[split] = data["meta"]
        if not data["X"]:
            datasets[split] = (np.empty((0, window, 1), dtype="float32"), np.empty((0,), dtype="float32"))
            continue
        X = np.asarray(data["X"], dtype="float32").reshape(-1, window, 1)
        y = np.asarray(data["y"], dtype="float32")
        datasets[split] = (X, y)

    return datasets, scalers, skipped, meta
