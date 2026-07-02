import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("bike_data.csv")

le_brand = LabelEncoder()
le_fuel = LabelEncoder()
le_seller = LabelEncoder()

df["Brand"] = le_brand.fit_transform(df["Brand"])
df["Fuel"] = le_fuel.fit_transform(df["Fuel"])
df["Seller"] = le_seller.fit_transform(df["Seller"])

X = df.drop("Price", axis=1)
y = df["Price"]

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(le_brand, "brand.pkl")
joblib.dump(le_fuel, "fuel.pkl")
joblib.dump(le_seller, "seller.pkl")

print("Model Created Successfully")