import os
from scraper import run_scraper

if __name__ == "__main__":
    # Scraper ausführen
    print("Starte Scraping-Prozess...")
    output_file = run_scraper(add_synthetic=True, num_synthetic=50)
    if output_file:
        print(f"Daten erfolgreich gespeichert in: {output_file}")
    else:
        print("Scraping fehlgeschlagen oder keine neuen Daten gefunden.")
