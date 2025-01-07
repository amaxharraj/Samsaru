import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def process_and_train(data_file="asgoodasnew_products.csv", model_file="linear_model.pkl"):
    print(f"Lade Daten aus {data_file}...")
    
    if not os.path.exists(data_file):
        print(f"Fehler: Die Datei {data_file} existiert nicht.")
        return None
    
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")
        return None

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['price'], inplace=True)

    if df.empty:
        print("Fehler: Die Daten sind leer!")
        return None

    df['variant_score'] = df['variant'].map({'neu': 1, 'wie neu': 0.8, 'sehr gut': 0.6, 'gut': 0.4})
    df['avg_price_by_title'] = df.groupby('title')['price'].transform('mean')
    df['price_deviation'] = df['price'] - df['avg_price_by_title']

    features = ['variant_score', 'price_deviation']
    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.path.exists(model_file):
        print(f"Lade vorhandenes Modell aus {model_file}...")
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            print("Erstelle neues Modell...")
            model = LinearRegression()
    else:
        print("Erstelle neues Modell...")
        model = LinearRegression()

    print("Starte Modelltraining...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Training abgeschlossen! R^2: {r2}")

    try:
        with open(model_file, 'wb') as f:
            pickle.dump({"model": model, "features": features, "r2": r2}, f)
        print(f"Modell erfolgreich in {os.path.abspath(model_file)} gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern des Modells: {e}")
        return None

    return model, features, df, r2
