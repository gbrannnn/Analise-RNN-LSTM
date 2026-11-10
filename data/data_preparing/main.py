import os
import re
import json
import unicodedata
from collections import Counter
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UF = "SP"
# Os dois primeiros dígitos do código IBGE do município identificam a UF (35 = SP).
UF_IBGE_PREFIX = "35"
RAW_DATA_DIR = os.path.join(BASE_DIR, "..", "raw_data", "pni", UF)
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "prepared_data", "pni", UF)
# Conjunto com variáveis exógenas agregadas por (município, vacina, mês). Fica numa
# pasta separada para não alterar o conjunto de série única consumido pelo baseline.
EXOGENOUS_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "prepared_data", "pni_exogenas", UF)
GENERATE_EXOGENOUS = True

# O layout do JSON da API mudou ao longo do tempo (e muda até dentro da mesma pasta
# de ano). Cada lista abaixo tem os nomes que um mesmo campo já teve, e o primeiro
# presente no registro é usado.
#   2020 - início/2021: st_documento, uf_estabelecimento
#   2021 - início/2022: st_documento, sg_uf_estabelecimento, sem codigo_vacina
#   2022 - início/2023: sem campo de status, traz registros de todas as UFs
#   2023 - início/2024: situacao_documento, município do estabelecimento só pelo nome
#   2024 em diante:     status_documento, sigla_uf_estabelecimento, todas as UFs
STATUS_FIELDS = ("st_documento", "situacao_documento", "status_documento")
UF_FIELDS = ("uf_estabelecimento", "sg_uf_estabelecimento", "sigla_uf_estabelecimento")
MUNICIPIO_NAME_FIELDS = ("nome_municipio_estabelecimento", "municipio_estabelecimento")
AGE_FIELDS = ("numero_idade_paciente", "idade_paciente")

# Faixas etárias (idade mínima, idade máxima inclusive, nome da coluna) usadas como
# proxy de público-alvo de campanha. A proporção de cada faixa no mês é calculada
# sobre as doses com idade válida.
AGE_BANDS = (
    (0, 1, "pct_idade_0_1"),
    (2, 4, "pct_idade_2_4"),
    (5, 19, "pct_idade_5_19"),
    (20, 59, "pct_idade_20_59"),
    (60, 200, "pct_idade_60_mais"),
)
MAX_VALID_AGE = 120

# codigo_natureza_estabelecimento "1" = ADMINISTRACAO PUBLICA. A descrição também é
# checada porque alguns registros trazem só um dos dois campos.
PUBLIC_NATUREZA_CODES = {"1"}

EXOGENOUS_COLUMNS = (
    ["idade_media"]
    + [name for _, _, name in AGE_BANDS]
    + ["pct_estabelecimento_publico", "pct_natureza_desconhecida"]
)

# Recorte temporal das séries (primeiro e último mês, inclusive). Fora dele a coleta
# bruta tem buracos (2021 sem codigo_vacina, 2023 só com jan/fev) que virariam
# zeros falsos na série, e 2026-09 foi coletado com o mês ainda em andamento.
PERIOD_START = datetime(2024, 1, 1).date()
PERIOD_END = datetime(2026, 8, 1).date()

# Série (município, vacina) é descartada se tiver menos que isso de meses com
# pelo menos 1 dose aplicada.
MIN_MONTHS_WITH_DATA = 12

# codigo_vacina do recorte de imunobiológicos críticos. Lista vazia mantém todas as vacinas.
# Os códigos vêm como string no JSON bruto (ex: "26"), por isso o set é de strings.
CRITICAL_VACCINE_CODES = {"15", "9", "33", "67"}


def find_raw_files(raw_dir, first_year, last_year):
    """
    Arquivos das pastas de ano entre first_year e last_year + 1.

    A pasta seguinte entra porque a coleta antiga gravava o fim de um ano no primeiro
    arquivo da pasta do ano seguinte. O recorte exato por mês é feito pelo data_vacina.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Diretório de dados brutos não encontrado: {raw_dir}")

    files = []
    for year in sorted(os.listdir(raw_dir)):
        year_dir = os.path.join(raw_dir, year)
        if not os.path.isdir(year_dir) or not year.isdigit():
            continue
        if not first_year <= int(year) <= last_year + 1:
            continue
        for filename in sorted(os.listdir(year_dir)):
            if filename.endswith(".json"):
                files.append(os.path.join(year_dir, filename))
    return files


def month_start(date):
    return date.replace(day=1)


def slugify(name):
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return normalized or "desconhecido"


def first_present(record, field_names):
    """Valor do primeiro campo de `field_names` que existe no registro (None se nenhum)."""
    for name in field_names:
        if name in record:
            return record[name]
    return None


def normalize_name(name):
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").strip().upper()


def parse_age(record):
    value = first_present(record, AGE_FIELDS)
    if value is None or value == "":
        return None
    try:
        age = int(float(value))
    except (TypeError, ValueError):
        return None
    if not 0 <= age <= MAX_VALID_AGE:
        return None
    return age


def is_public_establishment(record):
    """True/False para público/não público, None quando a natureza não vem no registro."""
    codigo = record.get("codigo_natureza_estabelecimento")
    descricao = record.get("descricao_natureza_estabelecimento")
    if codigo:
        return str(codigo) in PUBLIC_NATUREZA_CODES
    if descricao:
        return "PUBLICA" in normalize_name(descricao)
    return None


def accumulate_covariates(acc, record):
    """Soma no Counter `acc` os componentes das covariáveis de uma dose."""
    age = parse_age(record)
    if age is not None:
        acc["n_idade"] += 1
        acc["soma_idade"] += age
        for low, high, name in AGE_BANDS:
            if low <= age <= high:
                acc[name] += 1
                break

    publico = is_public_establishment(record)
    if publico is None:
        acc["n_natureza_desconhecida"] += 1
    else:
        acc["n_natureza"] += 1
        if publico:
            acc["n_publico"] += 1


def aggregate_monthly_counts(files, critical_vaccine_codes, collect_covariates=False):
    """
    Percorre os arquivos brutos e conta doses aplicadas por (município, vacina, mês).

    Só mantém em memória o contador agregado, nunca os registros brutos inteiros,
    já que cada arquivo bruto tem ~100 mil registros e o dataset completo passa de
    10 milhões de registros.

    Com `collect_covariates`, também acumula, na mesma passada e com os mesmos
    filtros, os componentes das variáveis exógenas de cada (município, vacina, mês).
    Elas são retornadas como 4º item; sem a flag, esse item é None.
    """
    counts = Counter()
    covariates = {} if collect_covariates else None
    covariates_by_municipio_name = {} if collect_covariates else None
    # Registros que só trazem o nome do município do estabelecimento (layout de
    # 2023/2024). São resolvidos para o código no fim, com os nomes vistos nos
    # registros que trazem código e nome juntos.
    counts_by_municipio_name = Counter()
    municipio_codes_by_name = {}
    municipio_names = {}
    vacina_name_counts = {}
    stats = Counter()

    for i, file_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] processando {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                pages = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  aviso: falha ao ler o arquivo ({exc}), pulando.")
            continue

        for page in pages:
            for record in page:
                stats["lidos"] += 1

                # Descarta rascunhos/registros provisórios e registros excluídos no
                # RNDS, senão contamos dose que nunca foi consolidada ou foi cancelada.
                # O layout de 2022/2023 não tem campo de status; nele, só o filtro de
                # exclusão se aplica.
                status = first_present(record, STATUS_FIELDS)
                has_status = any(name in record for name in STATUS_FIELDS)
                if (has_status and status != "final") or record.get("data_deletado_rnds"):
                    stats["qualidade"] += 1
                    continue

                codigo_municipio = record.get("codigo_municipio_estabelecimento")
                nome_municipio = first_present(record, MUNICIPIO_NAME_FIELDS)

                # Os layouts de 2022 em diante trazem registros de todas as UFs, mesmo
                # com o filtro uf_estabelecimento na requisição.
                if codigo_municipio:
                    in_uf = codigo_municipio.startswith(UF_IBGE_PREFIX)
                else:
                    in_uf = first_present(record, UF_FIELDS) == UF
                if not in_uf:
                    stats["fora_da_uf"] += 1
                    continue

                if "codigo_vacina" not in record:
                    stats["sem_codigo_vacina"] += 1
                    continue

                codigo_vacina = record.get("codigo_vacina")
                data_vacina = record.get("data_vacina")

                if (not codigo_municipio and not nome_municipio) or codigo_vacina is None or not data_vacina:
                    stats["faltante"] += 1
                    continue

                codigo_vacina = str(codigo_vacina)
                if critical_vaccine_codes and codigo_vacina not in critical_vaccine_codes:
                    stats["filtro_vacina"] += 1
                    continue

                try:
                    data = datetime.strptime(data_vacina[:10], "%Y-%m-%d").date()
                except ValueError:
                    stats["faltante"] += 1
                    continue

                if not PERIOD_START <= month_start(data) <= PERIOD_END:
                    stats["fora_do_periodo"] += 1
                    continue

                if codigo_municipio:
                    key = (codigo_municipio, codigo_vacina, month_start(data))
                    counts[key] += 1
                    target = covariates
                else:
                    key = (normalize_name(nome_municipio), codigo_vacina, month_start(data))
                    counts_by_municipio_name[key] += 1
                    target = covariates_by_municipio_name

                if collect_covariates:
                    accumulate_covariates(target.setdefault(key, Counter()), record)

                if codigo_municipio and nome_municipio:
                    municipio_codes_by_name.setdefault(normalize_name(nome_municipio), codigo_municipio)
                    municipio_names.setdefault(codigo_municipio, nome_municipio)

                # Layouts antigos não trazem o nome do município do estabelecimento, só
                # o código. Quando o paciente reside no mesmo município do estabelecimento,
                # usamos o nome do município do paciente como nome desse código.
                if codigo_municipio and codigo_municipio not in municipio_names:
                    codigo_paciente = record.get("codigo_municipio_paciente")
                    nome_paciente = record.get("nome_municipio_paciente")
                    if nome_paciente and codigo_paciente == codigo_municipio:
                        municipio_names[codigo_municipio] = nome_paciente
                        municipio_codes_by_name.setdefault(normalize_name(nome_paciente), codigo_municipio)

                # A API não retorna o nome da vacina, só o código. O fabricante mais
                # frequente daquele código vira só o rótulo legível da pasta de saída,
                # não é garantia de fabricante único por código.
                descricao_fabricante = record.get("descricao_vacina_fabricante")
                if descricao_fabricante:
                    vacina_name_counts.setdefault(codigo_vacina, Counter())[descricao_fabricante] += 1

    unresolved = Counter()
    for (nome, codigo_vacina, mes), total in counts_by_municipio_name.items():
        codigo_municipio = municipio_codes_by_name.get(nome)
        if codigo_municipio is None:
            unresolved[nome] += total
            continue
        counts[(codigo_municipio, codigo_vacina, mes)] += total
        if collect_covariates:
            acc = covariates.setdefault((codigo_municipio, codigo_vacina, mes), Counter())
            acc.update(covariates_by_municipio_name[(nome, codigo_vacina, mes)])

    vacina_names = {codigo: names.most_common(1)[0][0] for codigo, names in vacina_name_counts.items()}

    print(
        f"total de registros lidos: {stats['lidos']}, "
        f"descartados por qualidade (rascunho/excluído): {stats['qualidade']}, "
        f"fora da UF {UF}: {stats['fora_da_uf']}, "
        f"sem codigo_vacina no layout: {stats['sem_codigo_vacina']}, "
        f"descartados por dado faltante: {stats['faltante']}, "
        f"descartados pelo filtro de vacina: {stats['filtro_vacina']}, "
        f"fora do período {PERIOD_START:%Y-%m} a {PERIOD_END:%Y-%m}: {stats['fora_do_periodo']}"
    )
    if unresolved:
        print(
            f"aviso: {sum(unresolved.values())} doses com município só por nome sem código "
            f"correspondente, descartadas: {unresolved.most_common(10)}"
        )
    return counts, municipio_names, vacina_names, covariates


def build_series_dataframe(counts):
    rows = [
        {
            "codigo_municipio": codigo_municipio,
            "codigo_vacina": codigo_vacina,
            "mes": mes,
            "doses_aplicadas": total,
        }
        for (codigo_municipio, codigo_vacina, mes), total in counts.items()
    ]
    df = pd.DataFrame(rows, columns=["codigo_municipio", "codigo_vacina", "mes", "doses_aplicadas"])
    df["mes"] = pd.to_datetime(df["mes"])
    return df


def export_vacina_municipio_series(df, municipio_names, vacina_names, output_dir, min_months_with_data):
    os.makedirs(output_dir, exist_ok=True)

    # Índice temporal contíguo: meses sem nenhuma aplicação viram 0 explícito,
    # em vez de buraco na série (necessário pra LSTM). Fixo no período, para todas as
    # séries terem o mesmo intervalo mesmo que nenhum município tenha dose num extremo.
    full_range = pd.date_range(PERIOD_START, PERIOD_END, freq="MS")

    kept, discarded = [], []

    for (codigo_vacina, codigo_municipio), group in df.groupby(["codigo_vacina", "codigo_municipio"]):
        series = group.set_index("mes")["doses_aplicadas"].reindex(full_range, fill_value=0)
        months_with_data = int((series > 0).sum())

        nome_municipio = municipio_names.get(codigo_municipio, "desconhecido")
        nome_vacina = vacina_names.get(codigo_vacina, "desconhecido")
        entry = (codigo_vacina, nome_vacina, codigo_municipio, nome_municipio, months_with_data)

        if months_with_data < min_months_with_data:
            discarded.append(entry)
            continue

        vacina_dir = os.path.join(output_dir, f"{codigo_vacina}_{slugify(nome_vacina)}")
        os.makedirs(vacina_dir, exist_ok=True)

        out = series.rename("doses_aplicadas").rename_axis("mes").reset_index()
        out["mes"] = out["mes"].dt.strftime("%Y-%m-%d")

        filename = f"{codigo_municipio}_{slugify(nome_municipio)}.csv"
        out.to_csv(os.path.join(vacina_dir, filename), index=False)
        kept.append(entry)

    return kept, discarded


def build_exogenous_dataframe(counts, covariates):
    """
    Converte os acumuladores em covariáveis por (município, vacina, mês).

    Proporções são calculadas sobre as doses que trazem o campo correspondente
    (idade válida / natureza informada), para que registros sem o campo não puxem
    a proporção para baixo.
    """
    rows = []
    for (codigo_municipio, codigo_vacina, mes), total in counts.items():
        acc = covariates.get((codigo_municipio, codigo_vacina, mes), Counter())
        n_idade = acc["n_idade"]
        n_natureza = acc["n_natureza"]
        row = {
            "codigo_municipio": codigo_municipio,
            "codigo_vacina": codigo_vacina,
            "mes": mes,
            "doses_aplicadas": total,
            "idade_media": acc["soma_idade"] / n_idade if n_idade else float("nan"),
            "pct_estabelecimento_publico": acc["n_publico"] / n_natureza if n_natureza else float("nan"),
            "pct_natureza_desconhecida": acc["n_natureza_desconhecida"] / total if total else float("nan"),
        }
        for _, _, name in AGE_BANDS:
            row[name] = acc[name] / n_idade if n_idade else float("nan")
        rows.append(row)

    columns = ["codigo_municipio", "codigo_vacina", "mes", "doses_aplicadas"] + EXOGENOUS_COLUMNS
    df = pd.DataFrame(rows, columns=columns)
    df["mes"] = pd.to_datetime(df["mes"])
    return df


def export_exogenous_series(df, municipio_names, vacina_names, output_dir, min_months_with_data):
    """
    Mesma estrutura de pastas e mesmo critério de descarte do conjunto baseline,
    com as covariáveis como colunas extras.

    Meses sem aplicação recebem doses 0. Neles as covariáveis não existem (não há
    dose para tirar idade/natureza), então são preenchidas com o último valor
    conhecido da série (e, no início dela, com o próximo), e a coluna
    `mes_sem_aplicacao` marca esses meses para o modelo poder distingui-los.

    Não normaliza: a padronização por série é feita no treino (model/training/series.py)
    só com os meses de treino, para não vazar estatística de validação/teste.
    """
    os.makedirs(output_dir, exist_ok=True)
    full_range = pd.date_range(PERIOD_START, PERIOD_END, freq="MS")

    kept, discarded = [], []

    for (codigo_vacina, codigo_municipio), group in df.groupby(["codigo_vacina", "codigo_municipio"]):
        series = group.set_index("mes")[["doses_aplicadas"] + EXOGENOUS_COLUMNS].reindex(full_range)
        sem_aplicacao = series["doses_aplicadas"].isna()
        series["doses_aplicadas"] = series["doses_aplicadas"].fillna(0).astype("int64")
        months_with_data = int((series["doses_aplicadas"] > 0).sum())

        nome_municipio = municipio_names.get(codigo_municipio, "desconhecido")
        nome_vacina = vacina_names.get(codigo_vacina, "desconhecido")
        entry = (codigo_vacina, nome_vacina, codigo_municipio, nome_municipio, months_with_data)

        if months_with_data < min_months_with_data:
            discarded.append(entry)
            continue

        # Covariável ausente em toda a série (ex.: nenhum registro com idade) fica 0.
        series[EXOGENOUS_COLUMNS] = series[EXOGENOUS_COLUMNS].ffill().bfill().fillna(0.0).round(4)
        series["mes_sem_aplicacao"] = sem_aplicacao.astype("int64")

        vacina_dir = os.path.join(output_dir, f"{codigo_vacina}_{slugify(nome_vacina)}")
        os.makedirs(vacina_dir, exist_ok=True)

        out = series.rename_axis("mes").reset_index()
        out["mes"] = out["mes"].dt.strftime("%Y-%m-%d")

        filename = f"{codigo_municipio}_{slugify(nome_municipio)}.csv"
        out.to_csv(os.path.join(vacina_dir, filename), index=False)
        kept.append(entry)

    return kept, discarded


if __name__ == "__main__":
    raw_files = find_raw_files(RAW_DATA_DIR, PERIOD_START.year, PERIOD_END.year)
    print(f"{len(raw_files)} arquivos brutos encontrados em {RAW_DATA_DIR}")

    monthly_counts, municipio_names, vacina_names, monthly_covariates = aggregate_monthly_counts(
        raw_files, CRITICAL_VACCINE_CODES, collect_covariates=GENERATE_EXOGENOUS
    )
    series_df = build_series_dataframe(monthly_counts)

    kept, discarded = export_vacina_municipio_series(
        series_df, municipio_names, vacina_names, OUTPUT_DIR, MIN_MONTHS_WITH_DATA
    )

    if GENERATE_EXOGENOUS:
        exogenous_df = build_exogenous_dataframe(monthly_counts, monthly_covariates)
        exo_kept, _ = export_exogenous_series(
            exogenous_df, municipio_names, vacina_names, EXOGENOUS_OUTPUT_DIR, MIN_MONTHS_WITH_DATA
        )
        print(f"\nconjunto com variáveis exógenas: {len(exo_kept)} séries salvas em {EXOGENOUS_OUTPUT_DIR}")

    print(f"\nséries mantidas ({len(kept)}), salvas em {OUTPUT_DIR}:")
    for codigo_vacina, nome_vacina, codigo_municipio, nome_municipio, meses in sorted(kept, key=lambda x: -x[4]):
        print(f"  vacina {codigo_vacina} ({nome_vacina}) - {codigo_municipio} {nome_municipio}: {meses} meses com dado")

    print(f"\nséries descartadas por poucos dados ({len(discarded)}, < {MIN_MONTHS_WITH_DATA} meses):")
    for codigo_vacina, nome_vacina, codigo_municipio, nome_municipio, meses in sorted(discarded, key=lambda x: -x[4]):
        print(f"  vacina {codigo_vacina} ({nome_vacina}) - {codigo_municipio} {nome_municipio}: {meses} meses com dado")
