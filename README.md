# 📰 Fake News Detection Web App

An intelligent web application that detects whether a news article is **Real** or **Fake** using Natural Language Processing and Machine Learning. Built with 🐍 Python, ⚛️ Flask, and 🎨 Bootstrap.



## 🌟 Features

✅ Classify input news articles as **Real** or **Fake**  
✅ Clean and modern web interface using **HTML/CSS & Bootstrap**  
✅ Preprocessing with **NLTK stopwords and punctuation removal**  
✅ Trained using **TF-IDF** + **Logistic Regression**  
✅ Deploy-ready structure for platforms like Heroku, Render, Streamlit, etc.



## 🎥 Preview
![Screenshot 2025-05-31 131938](https://github.com/user-attachments/assets/bdf3ed6d-18d8-44ae-8f41-3b96b0efdadf)

![Screenshot 2025-05-31 131623](https://github.com/user-attachments/assets/4638016b-4227-4837-b9ce-fa91843cd8ce)

![Screenshot 2025-05-31 131803](https://github.com/user-attachments/assets/02ec7afb-2361-4b83-bc7e-2c4e65b89cc9)



## 📁 Project Structure

fake-news-app/
│
├── static/ # CSS and assets
│ └── style.css
├── templates/ # HTML templates
│ └── index.html
├── Dataset/ # Raw dataset (Fake.csv, True.csv)
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── app.py # Flask backend
├── train-model.py # Model training script


---

## 🛠 Tech Stack

- **Python 3.8+**
- **Flask** - lightweight web framework
- **NLTK** - text preprocessing
- **Scikit-learn** - model training
- **Bootstrap** - front-end styling

---

## 🧠 Dataset

This project uses the **"Fake and Real News Dataset"** from Kaggle.  
📦 Download it from: [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

- `Fake.csv` — contains fake news articles  
- `True.csv` — contains real news articles

---

## 🧪 How the Model Works

1. **Text Cleaning**: lowercasing, punctuation removal, stopwords removal  
2. **Vectorization**: using `TfidfVectorizer` to convert text to numerical format  
3. **Model**: `LogisticRegression` trained on a merged dataset of fake and real news  
4. **Evaluation**: high accuracy, precision, recall, and F1-score  

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone https://github.com/sujigunasekar/fake-news-detection.git
cd fake-news-app
```
## 2. Install dependencies
```
pip install flask pandas scikit-learn nltk
```
## 3. Train the model (optional if model.pkl already exists)
```
python train-model.py
```
## 4. Start the web application
```
python app.py
```
## 5. Visit in browser

Open http://127.0.0.1:5000 to use the app!

## 📦 requirements.txt

flask
pandas
scikit-learn
nltk

## ✅ Conclusion
This project successfully demonstrates the development of a machine learning-based Fake News Detection system using natural language processing techniques. By leveraging a combination of text preprocessing, TF-IDF vectorization, and Logistic Regression, the model was able to classify news articles with a high level of accuracy.

We trained the model on a labeled dataset of real and fake news articles and achieved promising results in terms of accuracy, precision, recall, and F1-score. Furthermore, we deployed the trained model using Flask, providing a simple yet effective web interface that allows users to input any news article and receive an instant prediction on its authenticity.

## 🔍 Key Learnings:
Text preprocessing (tokenization, stopword removal, punctuation stripping) is essential for working with natural language data.

TF-IDF is a powerful method for converting text into numerical features.

Logistic Regression can perform well in binary classification tasks like fake news detection.

Deploying ML models using Flask enables practical, user-facing applications.

## 🚀 Future Improvements:
Implement deep learning models (e.g., LSTM, BERT) for improved accuracy.

Include source credibility checks or fact-checking APIs.

Create a database to store user queries and feedback to improve the model iteratively.

Improve UI/UX with a more advanced and responsive frontend.

