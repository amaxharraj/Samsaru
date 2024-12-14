import os
import pandas as pd
from scrapy.crawler import CrawlerProcess
from used_console_spider import UsedConsolesSpider

def run_scraper(output_file="asgoodasnew_products.csv"):
    temp_file = "temp_scraped_data.csv"
    
    process = CrawlerProcess(settings={
        "FEEDS": {temp_file: {"format": "csv", "overwrite": True}},
        "LOG_LEVEL": "INFO"
    })
    process.crawl(UsedConsolesSpider)
    process.start()

    if os.path.exists(temp_file):
        new_data = pd.read_csv(temp_file)

        if os.path.exists(output_file):
            existing_data = pd.read_csv(output_file)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data.drop_duplicates(subset=["title", "variant"], keep="last", inplace=True)
        else:
            combined_data = new_data

        combined_data.to_csv(output_file, index=False)
        os.remove(temp_file)

        return output_file
    else:
        print("Keine neuen Daten gefunden.")
        return None
