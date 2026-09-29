import os
import re
import json
import unicodedata
from collections import Counter
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "..", "raw_data", "pni")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "prepared_data", "pni")

# Série (município, vacina) é descartada se tiver menos que isso de meses com
# pelo menos 1 dose aplicada.
MIN_MONTHS_WITH_DATA = 12

# codigo_vacina do recorte de imunobiológicos críticos. Lista vazia mantém todas as vacinas.
# Os códigos vêm como string no JSON bruto (ex: "26"), por isso o set é de strings.
CRITICAL_VACCINE_CODES = {"15", "9", "33", "67"}


def find_raw_files(raw_dir):
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Diretório de dados brutos não encontrado: {raw_dir}")

    files = []
    for year in sorted(os.listdir(raw_dir)):
        year_dir = os.path.join(raw_dir, year)
        if not os.path.isdir(year_dir):
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


def aggregate_monthly_counts(files, critical_vaccine_codes):
    """
    Percorre os arquivos brutos e conta doses aplicadas por (município, vacina, mês).

    Só mantém em memória o contador agregado, nunca os registros brutos inteiros,
    já que cada arquivo bruto tem ~100 mil registros e o dataset completo passa de
    10 milhões de registros.
    """
    counts = Counter()
    municipio_names = {}
    vacina_names = {}
    total_records = 0
    skipped_quality = 0
    skipped_missing = 0
    skipped_vaccine_filter = 0

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
                total_records += 1

                # Descarta rascunhos/registros provisórios e registros excluídos no
                # RNDS, senão contamos dose que nunca foi consolidada ou foi cancelada.
                if record.get("st_documento") != "final" or record.get("data_deletado_rnds"):
                    skipped_quality += 1
                    continue

                codigo_municipio = record.get("codigo_municipio_estabelecimento")
                codigo_vacina = record.get("codigo_vacina")
                data_vacina = record.get("data_vacina")

                if not codigo_municipio or codigo_vacina is None or not data_vacina:
                    skipped_missing += 1
                    continue

                if critical_vaccine_codes and codigo_vacina not in critical_vaccine_codes:
                    skipped_vaccine_filter += 1
                    continue

                try:
                    data = datetime.strptime(data_vacina[:10], "%Y-%m-%d").date()
                except ValueError:
                    skipped_missing += 1
                    continue

                counts[(codigo_municipio, codigo_vacina, month_start(data))] += 1

                # A API não retorna o nome do município do estabelecimento, só o código.
                # Quando o paciente reside no mesmo município do estabelecimento, usamos
                # o nome do município do paciente como nome desse código.
                if codigo_municipio not in municipio_names:
                    codigo_paciente = record.get("codigo_municipio_paciente")
                    nome_paciente = record.get("nome_municipio_paciente")
                    if nome_paciente and codigo_paciente == codigo_municipio:
                        municipio_names[codigo_municipio] = nome_paciente

                # A API não retorna o nome da vacina, só o código. Usamos o fabricante
                # do primeiro registro visto daquele código só como rótulo legível
                # da pasta de saída — não é garantia de fabricante único por código.
                if codigo_vacina not in vacina_names:
                    descricao_fabricante = record.get("descricao_vacina_fabricante")
                    if descricao_fabricante:
                        vacina_names[codigo_vacina] = descricao_fabricante

    print(
        f"total de registros lidos: {total_records}, "
        f"descartados por qualidade (rascunho/excluído): {skipped_quality}, "
        f"descartados por dado faltante: {skipped_missing}, "
        f"descartados pelo filtro de vacina: {skipped_vaccine_filter}"
    )
    return counts, municipio_names, vacina_names


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
    # em vez de buraco na série (necessário pra LSTM).
    full_range = pd.date_range(df["mes"].min(), df["mes"].max(), freq="MS")

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


if __name__ == "__main__":
    raw_files = find_raw_files(RAW_DATA_DIR)
    print(f"{len(raw_files)} arquivos brutos encontrados em {RAW_DATA_DIR}")

    monthly_counts, municipio_names, vacina_names = aggregate_monthly_counts(raw_files, CRITICAL_VACCINE_CODES)
    series_df = build_series_dataframe(monthly_counts)

    kept, discarded = export_vacina_municipio_series(
        series_df, municipio_names, vacina_names, OUTPUT_DIR, MIN_MONTHS_WITH_DATA
    )

    print(f"\nséries mantidas ({len(kept)}), salvas em {OUTPUT_DIR}:")
    for codigo_vacina, nome_vacina, codigo_municipio, nome_municipio, meses in sorted(kept, key=lambda x: -x[4]):
        print(f"  vacina {codigo_vacina} ({nome_vacina}) - {codigo_municipio} {nome_municipio}: {meses} meses com dado")

    print(f"\nséries descartadas por poucos dados ({len(discarded)}, < {MIN_MONTHS_WITH_DATA} meses):")
    for codigo_vacina, nome_vacina, codigo_municipio, nome_municipio, meses in sorted(discarded, key=lambda x: -x[4]):
        print(f"  vacina {codigo_vacina} ({nome_vacina}) - {codigo_municipio} {nome_municipio}: {meses} meses com dado")
