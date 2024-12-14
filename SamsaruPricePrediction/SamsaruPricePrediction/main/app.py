import os
from scraper import run_scraper
from model_training import process_and_train
from predictor import predict_price
import gradio as gr

def run_pipeline(title, variant):
    """Hauptpipeline: Scraping, Training, und Preisvorhersage."""
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
    predicted_price, optimized_price = predict_price(model, input_columns, df, title, variant)

    return predicted_price, optimized_price, round(avg_price, 2), round(r2, 4)

# Gradio Interface
interface = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Textbox(label="Titel der Konsole (z. B. Xbox Series X - 1TB schwarz)"),
        gr.Textbox(label="Zustand (z. B. wie neu, gut)")
    ],
    outputs=[
        gr.Textbox(label="Vorhergesagter Marktpreis"),
        gr.Textbox(label="Optimierter Verkaufspreis"),
        gr.Textbox(label="Durchschnittspreis aller Konsolen"),
        gr.Textbox(label="Modell R^2-Wert")
    ],
    title="Xbox-Konsolen-Preisoptimierer",
    description="Geben Sie den Konsolentitel und den Zustand ein, um Markt- und Verkaufspreise zu erhalten."
)

if __name__ == "__main__":
    interface.launch()