import pandas as pd

def get_dropdown_options(data_file):
    """Extrahiert Konsolenmodelle, Speicherkapazitäten und Zustände aus der CSV-Datei."""
    if not data_file or not pd.io.common.file_exists(data_file):
        return [], [], []

    df = pd.read_csv(data_file)

    # Modelle extrahieren (ohne Speicherkapazität)
    console_models = df['title'].str.extract(r'^(.*?)(?: - \d+[TG]B.*)?$')[0].dropna().unique().tolist()

    # Speicherkapazitäten extrahieren
    storage_options = df['title'].str.extract(r' - (\d+[TG]B)')[0].dropna().unique().tolist()

    # Zustände extrahieren
    conditions = df['variant'].dropna().unique().tolist()

    return sorted(console_models), sorted(storage_options), sorted(conditions)
