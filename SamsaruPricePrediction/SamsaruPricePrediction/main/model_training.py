import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def process_and_train(data_file="asgoodasnew_products.csv"):
    print(f"Lade Daten aus {data_file}...")
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Fehler beim Laden der CSV-Datei: {e}")
        return None

    # Bereinigung der Daten
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['price'], inplace=True)

    if df.empty:
        print("Die Daten sind leer!")
        return None

    df['variant_score'] = df['variant'].map({'neu': 1, 'wie neu': 0.8, 'sehr gut': 0.6, 'gut': 0.4})
    df['avg_price_by_title'] = df.groupby('title')['price'].transform('mean')
    df['price_deviation'] = df['price'] - df['avg_price_by_title']

    features = ['variant_score', 'price_deviation']
    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, enable_categorical=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print("Training abgeschlossen!")
    return model, features, df, r2
