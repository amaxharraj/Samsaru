import os
import pandas as pd
from scrapy.crawler import CrawlerProcess
from used_console_spider import UsedConsolesSpider
from synthetic_data import generate_synthetic_data

def run_scraper(output_file="asgoodasnew_products.csv", add_synthetic=True, num_synthetic=50):
    """
    Führt den Scraper aus und fügt optional synthetische Daten hinzu.
    :param output_file: Dateiname für die gespeicherten Daten.
    :param add_synthetic: Ob synthetische Daten hinzugefügt werden sollen.
    :param num_synthetic: Maximale Anzahl synthetischer Datenpunkte.
    """
    temp_file = "temp_scraped_data.csv"

    # Scrapy-Prozess starten
    process = CrawlerProcess(settings={
        "FEEDS": {temp_file: {"format": "csv", "overwrite": True}},
        "LOG_LEVEL": "INFO"
    })
    process.crawl(UsedConsolesSpider)
    process.start()

    if os.path.exists(temp_file):
        new_data = pd.read_csv(temp_file)
        
        if add_synthetic:
            print(f"Generiere maximal {num_synthetic} synthetische Datenpunkte...")
            synthetic_data = generate_synthetic_data(num_samples=num_synthetic)
            combined_data = pd.concat([new_data, synthetic_data], ignore_index=True)
        else:
            combined_data = new_data

        # Doppelte Einträge entfernen
        combined_data.drop_duplicates(subset=["title", "variant", "price", "scrape_date"], keep="last", inplace=True)
        
        # Daten speichern
        combined_data.to_csv(output_file, index=False)
        os.remove(temp_file)
        
        print(f"Daten erfolgreich in {output_file} gespeichert.")
        return output_file
    else:
        print("Keine neuen Daten gefunden.")
        return None
