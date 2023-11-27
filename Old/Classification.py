import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import make_pipeline

#nltk.download('stopwords')
#nltk.download('punkt')

df = pd.read_csv('./Datasets/amazon.csv')

# Datenbereinigung und Vorverarbeitung
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

df['processed_review_content'] = df['review_content'].apply(preprocess_text)

# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(df['processed_review_content'], df['rating'], test_size=0.2, random_state=42)

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