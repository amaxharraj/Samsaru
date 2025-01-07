import os
import pandas as pd
from scrapy.crawler import CrawlerProcess
from used_console_spider import UsedConsolesSpider
from synthetic_data import generate_synthetic_data  # Importiere die Funktion zur Generierung synthetischer Daten

def run_scraper(output_file="asgoodasnew_products.csv"):
    temp_file = "temp_scraped_data.csv"

    # Scrapy-Prozess starten
    process = CrawlerProcess(settings={
        "FEEDS": {temp_file: {"format": "csv", "overwrite": True}},
        "LOG_LEVEL": "INFO"
    })
    process.crawl(UsedConsolesSpider)
    process.start()

    combined_data = pd.DataFrame()

    if os.path.exists(temp_file):
        # Neue gescrappte Daten laden
        new_data = pd.read_csv(temp_file)

        # Existierende Daten laden (falls vorhanden)
        if os.path.exists(output_file):
            existing_data = pd.read_csv(output_file)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_data = new_data

        # Synthetische Daten generieren und hinzufügen
        print("Generiere synthetische Daten...")
        synthetic_data = generate_synthetic_data(1000)  # 1000 synthetische Datenpunkte
        combined_data = pd.concat([combined_data, synthetic_data], ignore_index=True)

        # Doppelte Einträge basierend auf Titel und Zustand entfernen
        combined_data.drop_duplicates(subset=["title", "variant", "price", "scrape_date"], keep="last", inplace=True)

        # Kombinierte Daten speichern
        combined_data.to_csv(output_file, index=False)

        # Temporäre Datei löschen
        os.remove(temp_file)

        print(f"Kombinierte Daten gespeichert in: {output_file}")
        return output_file
    else:
        print("Keine neuen Daten gefunden.")
        return None

