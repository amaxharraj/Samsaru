import pandas as pd
import numpy as np
from datetime import datetime

def generate_synthetic_data(num_samples=50):
    """Generiert synthetische Daten für Konsolen mit modellspezifischen Preisen."""
    variants = ['neu', 'wie neu', 'sehr gut', 'gut']
    titles = [
        'Microsoft Xbox Series X - 1TB schwarz',
        'Microsoft Xbox Series S - 1TB carbon black',
        'Microsoft Xbox Series S - 512GB weiß',
        'Microsoft Xbox One S - 500GB weiß'
    ]
    
    price_ranges = {
        'Microsoft Xbox Series X - 1TB schwarz': {'neu': (500, 600), 'wie neu': (450, 550), 'sehr gut': (400, 500), 'gut': (350, 450)},
        'Microsoft Xbox Series S - 1TB carbon black': {'neu': (400, 500), 'wie neu': (350, 450), 'sehr gut': (300, 400), 'gut': (250, 350)},
        'Microsoft Xbox Series S - 512GB weiß': {'neu': (350, 450), 'wie neu': (300, 400), 'sehr gut': (250, 350), 'gut': (200, 300)},
        'Microsoft Xbox One S - 500GB weiß': {'neu': (300, 400), 'wie neu': (250, 350), 'sehr gut': (200, 300), 'gut': (150, 250)}
    }
    
    synthetic_data = []
    for _ in range(num_samples):
        title = np.random.choice(titles)
        variant = np.random.choice(variants)
        price_min, price_max = price_ranges[title][variant]
        price = np.random.uniform(price_min, price_max)
        scrape_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        synthetic_data.append({
            'scrape_date': scrape_date,
            'title': title,
            'price': round(price, 2),
            'brand': 'Microsoft',
            'category': 'Konsolen' if 'Series' in title else 'Konsolen/Xbox',
            'variant': variant
        })

    return pd.DataFrame(synthetic_data)
