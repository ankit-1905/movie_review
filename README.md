# 🎬 Review Cinema — Review Sentiment Analyzer

A small web app that reads a product review and tells you whether it sounds **positive**, **negative** or **mixed** — with a confidence score and the words that influenced the result.

Built with **Python, scikit-learn and Streamlit**.

---

## ✨ Features

- **Single review mode** – paste a review, get a verdict, a star rating and a confidence percentage.
- **Influential words** – see which words pushed the prediction towards positive or negative.
- **Batch mode** – upload a CSV of reviews and download the results with sentiment for every row.
- **Recent history** – the last 20 reviews you analyzed in the session.
- **Movie-theatre themed UI** – dark cinema look with a ticket-style result card.

---

## 🧠 About the model

The model is a scikit-learn **Pipeline** saved as `review.joblib`. It has two steps:

| Step | Component | What it does |
|------|-----------|--------------|
| 1 | `TfidfVectorizer` | Turns the cleaned review text into numbers |
| 2 | `LogisticRegression` | Predicts positive or negative from those numbers |

### 1. Text cleaning (`wordopt`)
Before vectorizing, every review passes through a cleaning function that:

- converts text to lowercase
- removes text in square brackets, e.g. `[note]`
- removes links (`http://…`, `www…`) and HTML tags
- removes punctuation and special characters
- removes words that contain numbers

### 2. TF-IDF vectorizer
TF-IDF (*Term Frequency – Inverse Document Frequency*) gives each word a weight:

- words that appear often **in this review** get a higher weight
- words that appear in **almost every review** (like "the" or "is") get a lower weight

The vectorizer has a vocabulary of about **68,800 single words** (unigrams). No stop-word list is applied; TF-IDF weighting handles very common words.

### 3. Logistic Regression classifier
The classifier learns one weight per word:

- words like *great*, *love*, *excellent* get **positive** weights
- words like *waste*, *broken*, *terrible* get **negative** weights

The weights are added up and converted into a probability between 0 and 1.

| Output | Meaning |
|--------|---------|
| `1` | Positive review |
| `0` | Negative review |

Model settings: `max_iter = 20`.

---

## 📊 How to read the result

| Verdict | When it appears |
|---------|-----------------|
| 🍿 **Positive** (Blockbuster) | Positive probability above 60% |
| 🍅 **Negative** (Box-office flop) | Positive probability below 40% |
| 🎭 **Mixed** | Positive probability between 40% and 60% |

- **Confidence** is the probability of the winning side (e.g. 87% positive).
- **Stars** (★☆) are the positive probability scaled to a 1–5 rating.
- **Influential words** are the words in your review with the strongest pull towards each sentiment. Green chips push towards positive, red chips towards negative.

The 40–60% "Mixed" band can be changed with `NEUTRAL_BAND` at the top of `app.py`.

---

## 🚀 Run it locally

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

**2. Install dependencies**
```bash
pip install streamlit scikit-learn joblib pandas numpy
```

**3. Start the app**
```bash
streamlit run app.py
```

Then open the link shown in your terminal (usually http://localhost:8501).

### Project structure
```
├── app.py                  # Streamlit app
├── review.joblib           # Trained sentiment model
├── .streamlit/
│   └── config.toml         # Dark theme settings
└── README.md
```

> **Note:** `review.joblib` was saved with scikit-learn **1.9.1**. If your version is older, run `pip install -U scikit-learn`. The app also includes a small compatibility fix for version differences.

---

## 📂 Batch mode (CSV)

1. Open the **Batch (CSV)** tab.
2. Upload a CSV file that has a column containing reviews.
3. Choose that column and click **Run batch analysis**.
4. Download the results. Two columns are added: `sentiment` and `positive_probability`.

---

## ⚠️ Limitations

- It is a simple, fast model. It looks at **individual words**, not full meaning, so it can miss **sarcasm** ("Oh great, it broke in a day") and **negation** in long sentences.
- Very short reviews (one or two words) give low-confidence results.
- It works best on **English** product reviews similar to its training data.
- Scores close to 50% mean the model is unsure. Treat them as "mixed", not as a firm answer.

---

## 🔧 Possible improvements

- Add bigrams (`ngram_range=(1, 2)`) so phrases like "not good" are understood.
- Train for more iterations (`max_iter`) until the model fully converges.
- Try other models such as Linear SVM or a transformer-based model.
- Add a neutral class using 3-star reviews.

---

## 📝 Training details

> Fill in your own details here:
> - **Dataset:** _(name / source)_
> - **Number of reviews:** _(e.g. 50,000)_
> - **Train/test split:** _(e.g. 80/20)_
> - **Accuracy / F1-score:** _(e.g. 0.88)_

---

## 📄 License

Add your license here (e.g. MIT).
