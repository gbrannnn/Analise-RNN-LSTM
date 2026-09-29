"""
Análise de sensibilidade OFAT (um fator por vez) a partir do baseline.

Cada configuração é treinada com várias seeds, para separar o efeito da
perturbação do ruído de inicialização. Os resultados vão para o registro
acumulativo (artifacts/resultados.csv) e são consolidados em
artifacts/sensibilidade/.

Exemplos:
    python experiments.py                              # tudo
    python experiments.py --vacinas 9_serum --fatores dropout units
    python experiments.py --apenas-baseline --seeds 42 7 123
    python experiments.py --apenas-resumo              # só regera tabelas/gráficos
"""
import os
import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from series import find_vaccine_dirs, load_vaccine_series
from metrics import append_results
from lstm import (
    ARTIFACTS_DIR,
    PREPARED_DATA_DIR,
    RESULTS_PATH,
    ExperimentConfig,
    baseline_dir,
    run_experiment,
)

SUMMARY_DIR = os.path.join(ARTIFACTS_DIR, "sensibilidade")

BASELINE = ExperimentConfig()

SEEDS = [42, 7, 123, 2024, 31337]

# Valores testados por fator. O valor do baseline não precisa estar na lista:
# as rodadas de baseline já cobrem esse ponto de cada curva.
FACTORS = {
    "entrada": {
        # Com séries de 12 meses, val=2 e teste=2, a janela máxima é 7.
        "window": [2, 3, 6],
        "noise_level": [0.05, 0.1, 0.2, 0.3],
        "missing_ratio": [0.1, 0.2, 0.3],
    },
    "estrutural": {
        "dropout": [0.1, 0.2, 0.3],
        "n_layers": [2, 3],
        "units": [16, 32, 128],
        "learning_rate": [1e-4, 5e-4, 5e-3],
    },
}

METRICS = ["mae", "rmse", "mape", "mase"]

# Um efeito só é considerado relevante se a diferença para o baseline superar
# esse múltiplo do desvio-padrão entre seeds do próprio baseline.
NOISE_MULTIPLIER = 2.0

CONFIG_COLUMNS = [f.name for f in fields(ExperimentConfig) if f.name != "seed"]


def factor_group(fator):
    for grupo, factors in FACTORS.items():
        if fator in factors:
            return grupo
    raise KeyError(fator)


def planned_runs(baseline, fatores, seeds):
    """Lista de (grupo, fator, valor, config) na ordem de execução, começando pelo baseline."""
    runs = [("baseline", "baseline", np.nan, replace(baseline, seed=s)) for s in seeds]
    for fator in fatores:
        grupo = factor_group(fator)
        for valor in FACTORS[grupo][fator]:
            if valor == getattr(baseline, fator):
                continue
            for s in seeds:
                runs.append((grupo, fator, valor, replace(baseline, **{fator: valor}, seed=s)))
    return runs


def run_all(baseline, vacinas, fatores, seeds):
    execucao = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = planned_runs(baseline, fatores, seeds)

    for vaccine_dir in find_vaccine_dirs(PREPARED_DATA_DIR):
        nome_vacina = os.path.basename(vaccine_dir)
        if vacinas and nome_vacina not in vacinas:
            continue

        print(f"\n=== vacina {nome_vacina}: {len(runs)} treinos ===")
        clean_series = load_vaccine_series(vaccine_dir)

        for i, (grupo, fator, valor, config) in enumerate(runs, start=1):
            label = "baseline" if fator == "baseline" else f"{fator}={valor}"
            print(f"[{i}/{len(runs)}] {label} seed={config.seed}")

            # O modelo baseline da primeira seed é o que explain.py analisa.
            save_dir = baseline_dir(nome_vacina) if fator == "baseline" and config.seed == seeds[0] else None
            row = run_experiment(nome_vacina, clean_series, config, save_dir=save_dir)
            if row is None:
                continue

            row.update(
                execucao=execucao,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                grupo=grupo,
                fator=fator,
                valor=valor,
            )
            print(f"    MAE={row['mae']:.2f} RMSE={row['rmse']:.2f} MAPE={row['mape']:.2f}% MASE={row['mase']:.3f}")
            # Grava a cada treino para não perder o que já rodou se o processo for interrompido.
            append_results([row], RESULTS_PATH)


def matches_baseline(row, baseline):
    """True se a linha só difere do baseline no fator que ela varia."""
    base = asdict(baseline)
    for col in CONFIG_COLUMNS:
        if col == row["fator"]:
            continue
        if not np.isclose(float(row[col]), float(base[col])):
            return False
    return True


def load_registry(baseline):
    """
    Lê o registro acumulativo e mantém só as rodadas comparáveis ao baseline atual.

    Rodadas feitas com outro baseline (ex.: antes de mudar epochs) ficam de fora, e
    se a mesma configuração+seed foi rodada mais de uma vez vale a mais recente.
    """
    df = pd.read_csv(RESULTS_PATH)
    df = df[df.apply(matches_baseline, axis=1, baseline=baseline)]
    return df.drop_duplicates(subset=["vacina", "fator", "valor", "seed"], keep="last")


def summarize(baseline):
    if not os.path.exists(RESULTS_PATH):
        print(f"registro {RESULTS_PATH} não existe, nada para consolidar.")
        return

    df = load_registry(baseline)
    if df.empty:
        print("nenhuma rodada compatível com o baseline atual no registro.")
        return

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    base_rows = df[df["fator"] == "baseline"]
    agg = {f"{m}_{s}": (m, s) for m in METRICS for s in ("mean", "std")}

    base_stats = base_rows.groupby("vacina").agg(n_seeds=("seed", "nunique"), **agg)
    base_stats["baseline_aceito"] = base_stats["mase_mean"] < 1
    base_stats.to_csv(os.path.join(SUMMARY_DIR, "baseline_validacao.csv"))
    print("\nvalidação do baseline (aceito se MASE médio < 1):")
    print(base_stats[["n_seeds", "mase_mean", "mase_std", "mae_mean", "mape_mean", "baseline_aceito"]].to_string())

    # Cada curva de sensibilidade inclui o ponto do baseline no valor padrão do fator.
    curves = []
    for fator in df.loc[df["fator"] != "baseline", "fator"].unique():
        rows = df[df["fator"] == fator]
        base_point = base_rows.assign(
            fator=fator, grupo=factor_group(fator), valor=getattr(baseline, fator)
        )
        curves.append(pd.concat([rows, base_point]))

    if not curves:
        print("\nsó há rodadas de baseline no registro, sem curvas de sensibilidade.")
        return

    curves = pd.concat(curves)
    summary = (
        curves.groupby(["vacina", "grupo", "fator", "valor"])
        .agg(n_seeds=("seed", "nunique"), **agg)
        .reset_index()
    )

    for m in METRICS:
        base_mean = summary["vacina"].map(base_stats[f"{m}_mean"])
        base_std = summary["vacina"].map(base_stats[f"{m}_std"])
        summary[f"delta_{m}"] = summary[f"{m}_mean"] - base_mean
        summary[f"delta_{m}_pct"] = summary[f"delta_{m}"] / base_mean * 100
        summary[f"{m}_supera_ruido_seed"] = summary[f"delta_{m}"].abs() > NOISE_MULTIPLIER * base_std

    summary_path = os.path.join(SUMMARY_DIR, "resumo_sensibilidade.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nresumo salvo em {summary_path}")

    for fator, data in summary.groupby("fator"):
        plot_factor(fator, data, baseline)


def plot_factor(fator, data, baseline):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6))

    for ax, metric in zip(axes, METRICS):
        for vacina, rows in data.groupby("vacina"):
            rows = rows.sort_values("valor")
            ax.errorbar(
                rows["valor"],
                rows[f"{metric}_mean"],
                yerr=rows[f"{metric}_std"].fillna(0),
                marker="o",
                ms=4,
                capsize=3,
                label=vacina,
            )
        ax.axvline(getattr(baseline, fator), color="#888888", ls=":", lw=1)
        if metric == "mase":
            ax.axhline(1.0, color="#d62728", ls="--", lw=1)
        if fator == "learning_rate":
            ax.set_xscale("log")
        ax.set_title(metric.upper())
        ax.set_xlabel(fator)

    axes[0].legend(fontsize=8)
    fig.suptitle(f"Sensibilidade a {fator} (média ± desvio entre seeds; pontilhado = baseline)")
    fig.tight_layout()
    path = os.path.join(SUMMARY_DIR, f"sensibilidade_{fator}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args():
    all_factors = [f for factors in FACTORS.values() for f in factors]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vacinas", nargs="+", help="pastas de vacina a rodar (padrão: todas)")
    parser.add_argument("--fatores", nargs="+", choices=all_factors, default=all_factors)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, help="sobrescreve epochs do baseline (útil para testes rápidos)")
    parser.add_argument("--apenas-baseline", action="store_true", help="roda só o baseline em todas as seeds")
    parser.add_argument("--apenas-resumo", action="store_true", help="não treina, só consolida o registro")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    baseline = replace(BASELINE, epochs=args.epochs) if args.epochs else BASELINE

    if not args.apenas_resumo:
        fatores = [] if args.apenas_baseline else args.fatores
        run_all(baseline, args.vacinas, fatores, args.seeds)

    summarize(baseline)
