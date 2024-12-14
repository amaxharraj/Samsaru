import os
import pickle
import pandas as pd
import gradio as gr
from scraper import run_scraper
from model_training import process_and_train
from predictor import predict_price
from get_dropdown_options import get_dropdown_options

def run_pipeline(console_model, storage_capacity, condition):
    """Hauptpipeline: Vorhersagen basierend auf dem Modell."""
    print(f"Starte Vorhersage für Modell: {console_model}, Speicherkapazität: {storage_capacity}, Zustand: {condition}")

    # Überprüfen, ob das Modell existiert
    model_file = "model.pkl"
    if not os.path.exists(model_file):
        return "Fehler: Das Modell wurde noch nicht trainiert.", None, None, None

    # Modell laden
    print(f"Lade Modell aus {os.path.abspath(model_file)}...")
    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return f"Fehler beim Laden des Modells: {e}", None, None, None

    # Gescrappte Daten laden
    data_file = "asgoodasnew_products.csv"
    if not os.path.exists(data_file):
        return "Fehler: Keine gescrappten Daten verfügbar.", None, None, None

    df = pd.read_csv(data_file)

    # Flexibles Filtern nach Modell, Speicherkapazität und Zustand
    filtered_df = df[
        (df['title'].str.contains(console_model, case=False, na=False)) &
        (df['title'].str.contains(storage_capacity, case=False, na=False)) &
        (df['variant'] == condition)
    ]

    # Debugging-Ausgabe, falls keine Daten gefunden werden
    if filtered_df.empty:
        print(f"Keine passenden Daten gefunden für: {console_model}, {storage_capacity}, {condition}")
        print("Verfügbare Titel:")
        print(df['title'].unique())
        print("Verfügbare Varianten:")
        print(df['variant'].unique())
        return f"Keine Wettbewerbsdaten für '{condition}' verfügbar.", None, None, None

    # Vorhersagen generieren
    try:
        predicted_price, optimized_price = predict_price(
            model, ['variant_score', 'price_deviation'], filtered_df, console_model, condition
        )
        avg_price = filtered_df['price'].mean()
        r2_score_value = -1.012  # Beispielwert; passe an, falls nötig
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {e}")
        return f"Fehler bei der Vorhersage: {e}", None, None, None

    # Rückgabe der Ergebnisse
    return (
        predicted_price if predicted_price else "Keine Vorhersage verfügbar.",
        optimized_price if optimized_price else "Keine Optimierung verfügbar.",
        round(avg_price, 2) if not pd.isna(avg_price) else "Keine Durchschnittsdaten verfügbar.",
        round(r2_score_value, 4) if r2_score_value else "N/A"
    )


# Scraping ausführen und Dropdown-Werte laden
data_file = run_scraper()
console_models, storage_options, conditions = get_dropdown_options(data_file)

# Gradio Interface mit dynamischen Dropdown-Werten
interface = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Dropdown(label="Modell", choices=console_models, value=console_models[0] if console_models else None),
        gr.Dropdown(label="Speicherkapazität", choices=storage_options, value=storage_options[0] if storage_options else None),
        gr.Dropdown(label="Zustand", choices=conditions, value=conditions[0] if conditions else None),
    ],
    outputs=[
        gr.Textbox(label="Vorhergesagter Marktpreis"),
        gr.Textbox(label="Optimierter Verkaufspreis"),
        gr.Textbox(label="Durchschnittspreis aller Konsolen"),
        gr.Textbox(label="Modell R^2-Wert"),
    ],
    title="Konsolen-Preisoptimierer",
    description="Wählen Sie die Konsolenreihe, Speicherkapazität und den Zustand aus, um Markt- und Verkaufspreise zu erhalten.",
)

if __name__ == "__main__":
    # Modell prüfen und ggf. trainieren
    if not os.path.exists("model.pkl"):
        print("Trainiere Modell mit den vorhandenen Daten...")
        process_and_train()

    # Gradio-Interface starten
    interface.launch()
