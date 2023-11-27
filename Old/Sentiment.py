import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Annahme: Der Datensatz enthält Spalten "text" und "rating"
# "text" ist der Review-Text und "rating" ist die numerische Bewertung (0 bis 50)

# Laden des Datensatzes
df = pd.read_csv('./Datasets/amazon.csv')

#df_selected = df[['review_content', 'rating']]

# Datenbereinigung und Vorverarbeitung
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

df['processed_review_content'] = df['review_content'].apply(preprocess_text)

# Überprüfe den vorverarbeiteten Text
print(df['processed_review_content'].head())

# Konvertiere die 'rating'-Spalte in den Datentyp 'int'
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # 'coerce' wandelt nicht konvertierbare Werte in NaN um
# Kategorisierung der Ratings
df['sentiment'] = pd.cut(df['rating'], bins=[0, 1, 2, 3, 4, 5], labels=['sehr schlecht', 'schlecht', 'neutral', 'gut', 'sehr gut'], right=False)

df.dropna(inplace=True)
#print(df_selected.isna().sum())



# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(df['processed_review_content'], df['sentiment'], random_state=17)

# Erstellung des Klassifikationsmodells mit TF-IDF und Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Vorhersagen
y_pred = model.predict(X_test)

# Auswertung der Ergebnisse
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)
