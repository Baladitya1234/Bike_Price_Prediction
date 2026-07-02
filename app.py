from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
brand = joblib.load("brand.pkl")
fuel = joblib.load("fuel.pkl")
seller = joblib.load("seller.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    Brand = brand.transform([request.form['Brand']])[0]
    Year = int(request.form['Year'])
    Kms = int(request.form['Kms'])
    Owner = int(request.form['Owner'])
    Engine = int(request.form['Engine'])
    Fuel = fuel.transform([request.form['Fuel']])[0]
    Seller = seller.transform([request.form['Seller']])[0]

    data = pd.DataFrame([[Brand, Year, Kms, Owner, Engine, Fuel, Seller]],
                        columns=["Brand","Year","Kms_Driven","Owner","Engine","Fuel","Seller"])

    prediction = model.predict(data)[0]

    return render_template("index.html",
                           prediction_text=f"Estimated Bike Price: ₹ {prediction:,.0f}")


if __name__ == "__main__":
    app.run(debug=True)