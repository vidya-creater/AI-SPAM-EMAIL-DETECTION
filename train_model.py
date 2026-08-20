print("TRAIN MODEL STARTED")
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dataset load karo
data = pd.read_csv("spam.csv", encoding="latin-1")

# Dataset ke columns check karke required columns select karo
if "v1" in data.columns and "v2" in data.columns:
    data = data[["v1", "v2"]]
    data.columns = ["label", "message"]
elif "label" in data.columns and "message" in data.columns:
    data = data[["label", "message"]]
else:
    raise ValueError(
        "Dataset mein 'v1'/'v2' ya 'label'/'message' columns nahi mile."
    )

# Missing values remove karo
data.dropna(inplace=True)

# Spam = 1, Ham = 0
data["label"] = data["label"].map({"spam": 1, "ham": 0})

# Invalid rows remove
data.dropna(inplace=True)

# Input aur output
X = data["message"]
y = data["label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Model train
model.fit(X_train_tfidf, y_train)

# Test prediction
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("AI SPAM EMAIL DETECTION")
print("=" * 50)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# Model save karo
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel successfully trained!")
print("Created: spam_model.pkl")
print("Created: vectorizer.pkl")