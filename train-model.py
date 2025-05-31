import pandas as pd
import string
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# ✅ Use raw string (r"") to fix path issues on Windows
fake = pd.read_csv(r"C:\Users\SEC\Downloads\my-projects\fake-news-app\Dataset\Fake.csv.zip", encoding='ISO-8859-1', on_bad_lines='skip')
real = pd.read_csv(r"C:\Users\SEC\Downloads\my-projects\fake-news-app\Dataset\True.csv.zip", encoding='ISO-8859-1', on_bad_lines='skip')

# Add labels
fake['label'] = 0
real['label'] = 1

# Merge datasets
df = pd.concat([fake, real], axis=0)
df = df[['text', 'label']]
df = df.dropna(subset=['text'])  # Drop missing text rows just in case
df = df.sample(frac=1).reset_index(drop=True)

# Clean text
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved successfully!")
