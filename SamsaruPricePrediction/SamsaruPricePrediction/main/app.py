import os
import pickle
import pandas as pd
import gradio as gr
from scraper import run_scraper
from model_training import process_and_train
from predictor import predict_price
from get_dropdown_options import get_dropdown_options

def run_pipeline(console_model, storage_capacity, condition):
    print(f"Starte Vorhersage für Modell: {console_model}, Speicherkapazität: {storage_capacity}, Zustand: {condition}")

    model_file = "linear_model_with_regularization.pkl"
    if not os.path.exists(model_file):
        return "Fehler: Das Modell wurde noch nicht trainiert.", None, None, None

    try:
        with open(model_file, "rb") as f:
            model_data = pickle.load(f)
            model = model_data["model"]
            features = model_data["features"]
    except Exception as e:
        return f"Fehler beim Laden des Modells: {e}", None, None, None

    data_file = "asgoodasnew_products.csv"
    if not os.path.exists(data_file):
        return "Fehler: Keine gescrappten Daten verfügbar.", None, None, None

    df = pd.read_csv(data_file)

    filtered_df = df[
        (df['title'].str.contains(console_model, case=False, na=False)) &
        (df['title'].str.contains(storage_capacity, case=False, na=False)) &
        (df['variant'] == condition)
    ]

    if filtered_df.empty:
        return f"Keine Wettbewerbsdaten für '{condition}' verfügbar.", None, None, None

    try:
        predicted_price, optimized_price = predict_price(
            model, features, filtered_df, console_model, condition
        )
        avg_price = filtered_df['price'].mean()
    except Exception as e:
        return f"Fehler bei der Vorhersage: {e}", None, None, None

    return predicted_price, optimized_price, round(avg_price, 2)

if __name__ == "__main__":
    if not os.path.exists("linear_model_with_regularization.pkl"):
        print("Trainiere Modell mit den vorhandenen Daten...")
        process_and_train()

    data_file = run_scraper(add_synthetic=True, num_synthetic=50)
    console_models, storage_options, conditions = get_dropdown_options(data_file)

    # Gradio Interface
    with gr.Blocks(css="""
        button {background-color: #009999; color: white; border: none; font-weight: bold;}
    """) as interface:
        gr.Markdown("# **Samsaru-Verkaufspreisvorhersage**")
        gr.Markdown("### Finden Sie Verkaufspreise basierend auf aktuellen Daten.")
        
        with gr.Row():
            with gr.Column():
                model_input = gr.Dropdown(
                    label="Modell",
                    choices=console_models,
                    value=console_models[0] if console_models else None
                )
                storage_input = gr.Dropdown(
                    label="Speicherkapazität",
                    choices=storage_options,
                    value=storage_options[0] if storage_options else None
                )
                condition_input = gr.Dropdown(
                    label="Zustand",
                    choices=conditions,
                    value=conditions[0] if conditions else None
                )
                submit_button = gr.Button("Vorhersage starten")

            with gr.Column():
                predicted_price = gr.Textbox(label="Vorhergesagter Marktpreis")
                optimized_price = gr.Textbox(label="Optimierter Verkaufspreis")


        # Verbindung von Eingaben und Ausgaben mit der Funktion
        submit_button.click(
            fn=run_pipeline,
            inputs=[model_input, storage_input, condition_input],
            outputs=[predicted_price, optimized_price, avg_price]
        )
    
    interface.launch()
