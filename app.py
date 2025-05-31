from flask import Flask, render_template, request
import pickle
import string

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_text = request.form["news"]
        clean_text = preprocess(input_text)
        vector = vectorizer.transform([clean_text])
        pred = model.predict(vector)[0]
        prediction = "🟢 Real News" if pred == 1 else "🔴 Fake News"
    return render_template("index.html", prediction=prediction)

def preprocess(text):
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

if __name__ == "__main__":
    app.run(debug=True)
