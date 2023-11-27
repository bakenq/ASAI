import pandas as pd


# Laden des Datensatzes
df = pd.read_csv('./Datasets/amazon.csv')

# Überprüfe den Datentyp der 'rating'-Spalte
print(df['rating'].dtype)

# Konvertiere die 'rating'-Spalte in den Datentyp 'int'
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # 'coerce' wandelt nicht konvertierbare Werte in NaN um

# Überprüfe erneut den Datentyp der 'rating'-Spalte
print(df['rating'].dtype)
print(df['rating'].head())  # Zeigt die ersten 5 Zeilen


# Kategorisierung der Ratings
df['sentiment'] = pd.cut(df['rating'], bins=[-1, 1, 2, 3, 4, 5], labels=['sehr schlecht', 'schlecht', 'neutral', 'gut', 'sehr gut'])

# Du kannst die Anzahl der Bewertungen in jeder Kategorie überprüfen
print(df['sentiment'].value_counts())