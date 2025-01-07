import pandas as pd
import numpy as np
from datetime import datetime

def generate_synthetic_data(num_samples=1000):
    """Generiert synthetische Daten für Konsolen."""
    variants = ['neu', 'wie neu', 'sehr gut', 'gut']
    titles = [
        'Microsoft Xbox Series X - 1TB schwarz',
        'Microsoft Xbox Series S - 1TB carbon black',
        'Microsoft Xbox Series S - 512GB weiß',
        'Microsoft Xbox One S - 500GB weiß'
    ]
    prices = {
        'neu': np.random.uniform(350, 600, num_samples),
        'wie neu': np.random.uniform(300, 550, num_samples),
        'sehr gut': np.random.uniform(250, 500, num_samples),
        'gut': np.random.uniform(150, 400, num_samples)
    }
    synthetic_data = []

    for _ in range(num_samples):
        title = np.random.choice(titles)
        variant = np.random.choice(variants)
        price = np.random.choice(prices[variant])
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
