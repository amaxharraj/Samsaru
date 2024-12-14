import os
from scraper import run_scraper
from model_training import process_and_train
from predictor import predict_price
import gradio as gr

def run_pipeline(console_model, storage_capacity, condition):
    """Hauptpipeline: Scraping, Training, und Preisvorhersage."""
    # Kombiniere Modell und Speicherkapazität in einen Titel
    title = f"{console_model} - {storage_capacity}"

    print("Starte das Scraping...")
    scraped_file = run_scraper()

    if not scraped_file or not os.path.exists(scraped_file):
        print("Fehler: Scraping hat keine gültige Datei generiert.")
        return "Fehler beim Scraping.", None, None, None

    print("Starte das Training mit den gescrappten Daten...")
    result = process_and_train(scraped_file)

    if not result:
        print("Fehler: Das Training konnte nicht durchgeführt werden.")
        return "Fehler beim Training.", None, None, None

    model, input_columns, df, r2 = result

    if df.empty:
        print("Keine Daten für Vorhersagen verfügbar.")
        return "Keine Daten verfügbar.", None, None, None

    avg_price = df['price'].mean()
    predicted_price, optimized_price = predict_price(model, input_columns, df, title, condition)

    return predicted_price, optimized_price, round(avg_price, 2), round(r2, 4)

# Dropdown-Optionen
console_models = [
    "Xbox Series X",
    "Xbox Series S",
    "PlayStation 5",
]

storage_options = [
    "1TB",
    "512GB",
    "2TB",
]

conditions = [
    "wie neu",
    "gut",
    "akzeptabel",
    "defekt",
]

# Gradio Interface
interface = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Dropdown(label="Modell", choices=console_models, value="Xbox Series X"),
        gr.Dropdown(label="Speicherkapazität", choices=storage_options, value="1TB"),
        gr.Dropdown(label="Zustand", choices=conditions, value="wie neu"),
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
    interface.launch()
