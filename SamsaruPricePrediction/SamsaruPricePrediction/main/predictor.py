import pandas as pd

def predict_price(model, input_columns, df, title, variant):
    if 'title' not in df.columns or 'variant' not in df.columns:
        raise KeyError("Die benötigten Spalten 'title' und/oder 'variant' fehlen im DataFrame!")

    avg_competitor_price = df[ 
        (df['title'].str.contains(title, case=False, na=False)) & 
        (df['variant'] == variant)
    ]['price'].mean()

    if pd.isna(avg_competitor_price):
        return f"Keine Wettbewerbsdaten für '{variant}' verfügbar.", None

    variant_score = {'neu': 1, 'wie neu': 0.8, 'sehr gut': 0.6, 'gut': 0.4}.get(variant.lower(), 0.4)

    # Korrektur: Alle notwendigen Features müssen übergeben werden
    input_data = pd.DataFrame([{
        'variant_score': variant_score,
        'price_deviation': 0,  # Preisabweichung basierend auf der Vorhersage
        'brand': 0,  # Standardmarke (oder falls bekannt, spezifisch anpassen)
        'category': 0  # Standardkategorie (oder anpassen)
    }])

    prediction = model.predict(input_data)[0]
    optimized_price = max(0.9 * avg_competitor_price, prediction * 0.95)

    return round(prediction, 2), round(optimized_price, 2)
