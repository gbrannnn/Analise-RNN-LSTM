import os
import re
import json
import unicodedata
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "..", "raw_data", "pni")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "prepared_data", "pni")

# Município é descartado se tiver menos que isso de semanas com pelo menos 1 dose aplicada.
MIN_WEEKS_WITH_DATA = 52


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


def week_start(date):
    return date - timedelta(days=date.weekday())


def slugify(name):
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return normalized or "desconhecido"


def aggregate_weekly_counts(files):
    """
    Percorre os arquivos brutos e conta doses aplicadas por (município, semana).

    Só mantém em memória o contador agregado, nunca os registros brutos inteiros,
    já que cada arquivo bruto tem ~100 mil registros e o dataset completo passa de
    10 milhões de registros.
    """
    counts = Counter()
    municipio_names = {}
    total_records = 0
    skipped_records = 0

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

                codigo_estabelecimento = record.get("codigo_municipio_estabelecimento")
                data_vacina = record.get("data_vacina")

                if not codigo_estabelecimento or not data_vacina:
                    skipped_records += 1
                    continue

                try:
                    data = datetime.strptime(data_vacina[:10], "%Y-%m-%d").date()
                except ValueError:
                    skipped_records += 1
                    continue

                counts[(codigo_estabelecimento, week_start(data))] += 1

                # A API não retorna o nome do município do estabelecimento, só o código.
                # Quando o paciente reside no mesmo município do estabelecimento, usamos
                # o nome do município do paciente como nome desse código.
                if codigo_estabelecimento not in municipio_names:
                    codigo_paciente = record.get("codigo_municipio_paciente")
                    nome_paciente = record.get("nome_municipio_paciente")
                    if nome_paciente and codigo_paciente == codigo_estabelecimento:
                        municipio_names[codigo_estabelecimento] = nome_paciente

    print(f"total de registros lidos: {total_records}, ignorados (sem município/data): {skipped_records}")
    return counts, municipio_names


def build_series_dataframe(counts):
    rows = [
        {"codigo_municipio": codigo, "semana": semana, "doses_aplicadas": total}
        for (codigo, semana), total in counts.items()
    ]
    df = pd.DataFrame(rows, columns=["codigo_municipio", "semana", "doses_aplicadas"])
    df["semana"] = pd.to_datetime(df["semana"])
    return df


def export_municipio_series(df, municipio_names, output_dir, min_weeks_with_data):
    os.makedirs(output_dir, exist_ok=True)

    full_range = pd.date_range(df["semana"].min(), df["semana"].max(), freq="W-MON")

    kept, discarded = [], []

    for codigo_municipio, group in df.groupby("codigo_municipio"):
        series = group.set_index("semana")["doses_aplicadas"].reindex(full_range, fill_value=0)
        weeks_with_data = int((series > 0).sum())
        nome_municipio = municipio_names.get(codigo_municipio, "desconhecido")

        if weeks_with_data < min_weeks_with_data:
            discarded.append((codigo_municipio, nome_municipio, weeks_with_data))
            continue

        out = series.rename("doses_aplicadas").rename_axis("semana").reset_index()
        out["semana"] = out["semana"].dt.strftime("%Y-%m-%d")

        filename = f"{codigo_municipio}_{slugify(nome_municipio)}.csv"
        out.to_csv(os.path.join(output_dir, filename), index=False)
        kept.append((codigo_municipio, nome_municipio, weeks_with_data))

    return kept, discarded


if __name__ == "__main__":
    raw_files = find_raw_files(RAW_DATA_DIR)
    print(f"{len(raw_files)} arquivos brutos encontrados em {RAW_DATA_DIR}")

    weekly_counts, municipio_names = aggregate_weekly_counts(raw_files)
    series_df = build_series_dataframe(weekly_counts)

    kept, discarded = export_municipio_series(series_df, municipio_names, OUTPUT_DIR, MIN_WEEKS_WITH_DATA)

    print(f"\nmunicípios mantidos ({len(kept)}), salvos em {OUTPUT_DIR}:")
    for codigo, nome, semanas in sorted(kept, key=lambda x: -x[2]):
        print(f"  {codigo} - {nome}: {semanas} semanas com dado")

    print(f"\nmunicípios descartados por poucos dados ({len(discarded)}, < {MIN_WEEKS_WITH_DATA} semanas):")
    for codigo, nome, semanas in sorted(discarded, key=lambda x: -x[2]):
        print(f"  {codigo} - {nome}: {semanas} semanas com dado")
