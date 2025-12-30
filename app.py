from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["MedInc"]),
            float(request.form["HouseAge"]),
            float(request.form["AveRooms"]),
            float(request.form["AveBedrms"]),
            float(request.form["Population"]),
            float(request.form["AveOccup"]),
            float(request.form["Latitude"]),
            float(request.form["Longitude"]),
        ]

        final_features = scaler.transform([features])
        prediction = model.predict(final_features)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
