from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Trained model aur vectorizer load karo
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email", "").strip()

        if email_text:
            # Email ko TF-IDF mein convert karo
            email_vector = vectorizer.transform([email_text])

            # Prediction
            prediction = model.predict(email_vector)[0]

            # Probability
            probabilities = model.predict_proba(email_vector)[0]
            confidence = round(max(probabilities) * 100, 2)

            if prediction == 1:
                result = "SPAM"
            else:
                result = "NOT SPAM"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        email_text=email_text
    )


if __name__ == "__main__":
    print("FLASK APP STARTING...")
    app.run(debug=True)