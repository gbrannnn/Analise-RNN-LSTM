"""
Explicabilidade (SHAP e LIME) do modelo baseline já treinado de cada vacina.

Cada mês da janela de entrada é tratado como uma feature tabular ("t-12" ... "t-1",
onde t-1 é o mês imediatamente anterior ao previsto). As importâncias saem na
escala normalizada do modelo, que é comum a todos os municípios.

Requer o baseline salvo em artifacts/<vacina>/baseline (rode lstm.py ou experiments.py).

Com --dados exogenas, explica o modelo de artifacts/<vacina>/exogenas. Cada feature
é (variável, lag), por exemplo "idade_media t-3". Além dos CSVs completos, as
importâncias são agregadas por variável (somando os lags) e por lag (somando as
variáveis).

Exemplos:
    python explain.py
    python explain.py --vacinas 9_serum --conjunto todos --max-amostras 500
    python explain.py --dados exogenas
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

from series import build_supervised_dataset, find_vaccine_dirs, input_feature_names, perturb_series
from lstm import DATASETS, add_dataset_arg, baseline_dir, load_dataset, load_saved_model


def lag_names(window):
    return [f"t-{window - i}" for i in range(window)]


def lag_feature_names(window, variables):
    """Nomes na ordem de X achatado (lag maior, variável menor): 'doses_aplicadas t-12', ..."""
    return [f"{variable} {lag}" for lag in lag_names(window) for variable in variables]


def select_samples(datasets, meta, conjunto, max_amostras, seed):
    splits = ["test"] if conjunto == "teste" else ["train", "val", "test"]
    X = np.concatenate([datasets[s][0] for s in splits])
    info = [m for s in splits for m in meta[s]]

    if len(X) > max_amostras:
        idx = np.sort(np.random.default_rng(seed).choice(len(X), size=max_amostras, replace=False))
        X = X[idx]
        info = [info[i] for i in idx]

    info = pd.DataFrame(info, columns=["municipio", "mes_alvo"])
    return X.reshape(len(X), -1), info


def explain_shap(predict_fn, X_background, X_explain, n_background, **shap_kwargs):
    """`shap_kwargs` vai para KernelExplainer.shap_values (ex.: nsamples, l1_reg); vazio mantém o padrão do SHAP."""
    background = shap.kmeans(X_background, min(n_background, len(X_background)))
    explainer = shap.KernelExplainer(predict_fn, background)
    return np.asarray(explainer.shap_values(X_explain, silent=True, **shap_kwargs)).reshape(X_explain.shape)


def explain_lime(predict_fn, X_background, X_explain, names, seed, num_samples):
    explainer = LimeTabularExplainer(
        X_background,
        mode="regression",
        feature_names=names,
        # Sem discretização o peso é um coeficiente linear local por lag, comparável ao SHAP.
        discretize_continuous=False,
        random_state=seed,
    )
    weights = np.zeros_like(X_explain)
    for i, x in enumerate(X_explain):
        exp = explainer.explain_instance(x, predict_fn, num_features=len(names), num_samples=num_samples)
        for feature_idx, weight in next(iter(exp.as_map().values())):
            weights[i, feature_idx] = weight
    return weights


def save_bar(values, names, title, path):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(names)), 3.5))
    ax.bar(names, values, color="#1f77b4")
    ax.set_ylabel("importância média |valor|")
    ax.set_title(title)
    if max(len(n) for n in names) > 5:
        ax.tick_params(axis="x", labelrotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_heatmap_by_month(values, info, names, title, path):
    """Importância média |valor| por mês do calendário previsto x lag da janela."""
    df = pd.DataFrame(np.abs(values), columns=names)
    df["mes"] = info["mes_alvo"].dt.month.to_numpy()
    table = df.groupby("mes")[names].mean()

    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 0.45 * len(table) + 1.8))
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(table)), [f"{m:02d}" for m in table.index])
    ax.set_xlabel("mês da janela de entrada")
    ax.set_ylabel("mês previsto")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return table


def aggregate_importance(values, window, variables):
    """
    |valor| médio por (lag, variável) somado em cada eixo.

    Retorna (por_variavel, por_lag, por_amostra_e_lag), este último usado no mapa
    por mês previsto.
    """
    abs_values = np.abs(values).reshape(len(values), window, len(variables))
    mean_abs = abs_values.mean(axis=0)
    return mean_abs.sum(axis=0), mean_abs.sum(axis=1), abs_values.sum(axis=2)


def save_comparison(shap_importance, lime_importance, labels, label_name, title, csv_path, png_path):
    """Importâncias SHAP x LIME normalizadas para somar 1 (as escalas dos dois diferem)."""
    comparison = pd.DataFrame(
        {
            label_name: labels,
            "shap_importancia": shap_importance,
            "lime_importancia": lime_importance,
            "shap_normalizada": shap_importance / shap_importance.sum(),
            "lime_normalizada": lime_importance / lime_importance.sum(),
        }
    )
    comparison.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(max(6.5, 0.7 * len(labels)), 3.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, comparison["shap_normalizada"], width=0.4, label="SHAP")
    ax.bar(x + 0.2, comparison["lime_normalizada"], width=0.4, label="LIME")
    ax.set_xticks(x, labels)
    if max(len(n) for n in labels) > 5:
        ax.tick_params(axis="x", labelrotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
    ax.set_ylabel("importância relativa")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return comparison


def save_exogenous_outputs(shap_values, lime_weights, info, window, variables, nome_vacina, output_dir):
    """Gráficos e tabelas agregados por variável e por lag do modelo com covariáveis."""
    lags = lag_names(window)
    shap_var, shap_lag, shap_sample_lag = aggregate_importance(shap_values, window, variables)
    lime_var, lime_lag, _ = aggregate_importance(lime_weights, window, variables)

    save_bar(shap_var, variables, f"SHAP por variável - {nome_vacina}", os.path.join(output_dir, "shap_importancia_variavel.png"))
    save_bar(shap_lag, lags, f"SHAP por lag - {nome_vacina}", os.path.join(output_dir, "shap_importancia.png"))
    save_bar(lime_var, variables, f"LIME por variável - {nome_vacina}", os.path.join(output_dir, "lime_importancia_variavel.png"))
    save_bar(lime_lag, lags, f"LIME por lag - {nome_vacina}", os.path.join(output_dir, "lime_importancia.png"))

    save_heatmap_by_month(
        shap_sample_lag, info, lags, f"|SHAP| por mês previsto (soma das variáveis) - {nome_vacina}",
        os.path.join(output_dir, "shap_por_mes.png"),
    ).to_csv(os.path.join(output_dir, "shap_por_mes.csv"))

    by_variable = save_comparison(
        shap_var, lime_var, variables, "variavel", f"SHAP x LIME por variável - {nome_vacina}",
        os.path.join(output_dir, "comparacao_shap_lime_variavel.csv"),
        os.path.join(output_dir, "comparacao_shap_lime_variavel.png"),
    )
    save_comparison(
        shap_lag, lime_lag, lags, "lag", f"SHAP x LIME por lag - {nome_vacina}",
        os.path.join(output_dir, "comparacao_shap_lime.csv"),
        os.path.join(output_dir, "comparacao_shap_lime.png"),
    )
    print(by_variable[["variavel", "shap_normalizada", "lime_normalizada"]].to_string(index=False))


def explain_vaccine(vaccine_dir, args):
    nome_vacina = os.path.basename(vaccine_dir)
    model_dir = baseline_dir(nome_vacina, args.dados)
    if not os.path.exists(os.path.join(model_dir, "modelo.keras")):
        print(f"{nome_vacina}: modelo não encontrado em {model_dir}, pulando.")
        return

    print(f"\n=== vacina {nome_vacina} ({args.dados}) ===")
    model, config = load_saved_model(model_dir)

    clean_series, covariates = load_dataset(vaccine_dir, args.dados)
    model_series = perturb_series(
        clean_series, config.noise_level, config.missing_ratio, config.val_months, config.test_months, config.seed
    )
    datasets, _, _, meta = build_supervised_dataset(
        model_series, config.window, config.val_months, config.test_months, covariates
    )

    window = config.window
    variables = input_feature_names(covariates is not None)
    n_features = len(variables)
    # No baseline (só doses) os nomes continuam "t-12" ... "t-1".
    names = lag_names(window) if n_features == 1 else lag_feature_names(window, variables)
    X_train = datasets["train"][0].reshape(-1, window * n_features)
    X_explain, info = select_samples(datasets, meta, args.conjunto, args.max_amostras, config.seed)
    print(f"explicando {len(X_explain)} amostras ({args.conjunto}), fundo com {len(X_train)} amostras de treino")

    def predict_fn(x):
        return model.predict(np.asarray(x, dtype="float32").reshape(-1, window, n_features), verbose=0).reshape(-1)

    output_dir = os.path.join(model_dir, "explicabilidade")
    os.makedirs(output_dir, exist_ok=True)

    if n_features > 1:
        explain_exogenous(predict_fn, X_train, X_explain, info, names, window, variables, nome_vacina, config, args, output_dir)
        return

    print("SHAP (KernelExplainer)...")
    shap_values = explain_shap(predict_fn, X_train, X_explain, args.fundo)
    pd.concat([info, pd.DataFrame(shap_values, columns=names)], axis=1).to_csv(
        os.path.join(output_dir, "shap_valores.csv"), index=False
    )
    shap_importance = np.abs(shap_values).mean(axis=0)
    save_bar(shap_importance, names, f"SHAP - {nome_vacina}", os.path.join(output_dir, "shap_importancia.png"))

    shap.summary_plot(shap_values, X_explain, feature_names=names, show=False)
    plt.title(f"SHAP por amostra - {nome_vacina}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=120)
    plt.close("all")

    save_heatmap_by_month(
        shap_values, info, names, f"|SHAP| por mês previsto - {nome_vacina}",
        os.path.join(output_dir, "shap_por_mes.png"),
    ).to_csv(os.path.join(output_dir, "shap_por_mes.csv"))

    print("LIME (LimeTabularExplainer)...")
    lime_weights = explain_lime(predict_fn, X_train, X_explain, names, config.seed, args.lime_amostras)
    pd.concat([info, pd.DataFrame(lime_weights, columns=names)], axis=1).to_csv(
        os.path.join(output_dir, "lime_pesos.csv"), index=False
    )
    lime_importance = np.abs(lime_weights).mean(axis=0)
    save_bar(lime_importance, names, f"LIME - {nome_vacina}", os.path.join(output_dir, "lime_importancia.png"))

    # Normalizadas para somar 1, já que SHAP e LIME têm escalas diferentes.
    comparison = pd.DataFrame(
        {
            "lag": names,
            "shap_importancia": shap_importance,
            "lime_importancia": lime_importance,
            "shap_normalizada": shap_importance / shap_importance.sum(),
            "lime_normalizada": lime_importance / lime_importance.sum(),
        }
    )
    comparison.to_csv(os.path.join(output_dir, "comparacao_shap_lime.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(window)
    ax.bar(x - 0.2, comparison["shap_normalizada"], width=0.4, label="SHAP")
    ax.bar(x + 0.2, comparison["lime_normalizada"], width=0.4, label="LIME")
    ax.set_xticks(x, names)
    ax.set_ylabel("importância relativa")
    ax.set_title(f"SHAP x LIME - {nome_vacina}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparacao_shap_lime.png"), dpi=120)
    plt.close(fig)

    print(comparison[["lag", "shap_normalizada", "lime_normalizada"]].to_string(index=False))
    print(f"artefatos salvos em {output_dir}")


def explain_exogenous(predict_fn, X_train, X_explain, info, names, window, variables, nome_vacina, config, args, output_dir):
    """SHAP e LIME do modelo com covariáveis: CSVs com as window x variáveis features e gráficos agregados."""
    # Com window x variáveis features, o l1_reg="auto" padrão faz seleção de features
    # e zera as de efeito menor (as covariáveis saíam todas 0). Sem regularização, todas
    # recebem valor, e mais amostras de coalizão compensam a estimativa sem seleção.
    print(f"SHAP (KernelExplainer) em {len(names)} features, nsamples={args.shap_amostras}, sem l1_reg...")
    shap_values = explain_shap(
        predict_fn, X_train, X_explain, args.fundo, nsamples=args.shap_amostras, l1_reg=False
    )
    pd.concat([info, pd.DataFrame(shap_values, columns=names)], axis=1).to_csv(
        os.path.join(output_dir, "shap_valores.csv"), index=False
    )

    shap.summary_plot(shap_values, X_explain, feature_names=names, show=False)
    plt.title(f"SHAP por amostra - {nome_vacina}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=120)
    plt.close("all")

    print("LIME (LimeTabularExplainer)...")
    lime_weights = explain_lime(predict_fn, X_train, X_explain, names, config.seed, args.lime_amostras)
    pd.concat([info, pd.DataFrame(lime_weights, columns=names)], axis=1).to_csv(
        os.path.join(output_dir, "lime_pesos.csv"), index=False
    )

    save_exogenous_outputs(shap_values, lime_weights, info, window, variables, nome_vacina, output_dir)
    print(f"artefatos salvos em {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vacinas", nargs="+", help="pastas de vacina (padrão: todas com baseline salvo)")
    parser.add_argument(
        "--conjunto", choices=["teste", "todos"], default="teste",
        help="amostras a explicar; 'todos' cobre mais meses do calendário no mapa por mês",
    )
    parser.add_argument("--max-amostras", type=int, default=200)
    parser.add_argument("--fundo", type=int, default=20, help="centroides k-means do fundo do SHAP")
    parser.add_argument("--lime-amostras", type=int, default=1000, help="perturbações por instância no LIME")
    parser.add_argument(
        "--shap-amostras", type=int, default=5000,
        help="nsamples do KernelExplainer, só no fluxo --dados exogenas (o baseline usa o padrão do SHAP)",
    )
    add_dataset_arg(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for vaccine_dir in find_vaccine_dirs(DATASETS[args.dados]["prepared_dir"]):
        if args.vacinas and os.path.basename(vaccine_dir) not in args.vacinas:
            continue
        explain_vaccine(vaccine_dir, args)
