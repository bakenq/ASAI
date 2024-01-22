# Setup

#### **Referenz für Ordner-Struktur**
![alt text](https://i.ibb.co/gjnm6wp/Folder-Structure.jpg)

## **Datasets**
Datasets Ordner ist nicht im Repo enthalten!   
Lokale Erstellung von "Datasets"-Ordner benötigt (siehe Referenz für Ordner-Struktur).  
[Kaggle: Amazon Reviews (Cellphones and Accessories)](https://www.kaggle.com/datasets/abdallahwagih/amazon-reviews/data)  
[Kaggle: Amazon Reviews (Books)](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/) 

## **Models**
Models Ordner ist ebenfalls nicht im Repo enthalten!  
Das benötigte Model für das Web-Interface ("DistilBERT_Model") kann [hier](https://drive.google.com/drive/folders/1k4e39e95D6tx5J9VwOpA45a3I5BZZTe_?usp=sharing) runtergeladen werden.

## **Wichtige Anmerkungen**
Ein Großteil der Notebooks verwendet eine modifizierte/vorverarbeitete Version der Datensätze. Diese werden in **Dataset_phones_preprocessing** und **Dataset_books_preprocessing** erstellt. Führt diese Notebooks also als erste aus!  
Desweiteren sind einige Notebooks über Google Colab ausgeführt worden und können nicht so einfach lokal zum Laufen gebracht werden. Diese Notebooks sind mit **"Colab"** im Namen gekennzeichnet.

#### **Needed/Useful Installs**
pandas  
nltk  
scikit-learn  
matplotlib  
seaborn  
tensorflow  
flask  
(und vielleicht mehr)