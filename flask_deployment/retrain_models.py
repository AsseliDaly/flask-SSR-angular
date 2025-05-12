import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def retrain_hotel_model():
    """Retrain and save hotel price prediction model"""
    print("Training hotel model...")
    df = pd.read_csv("data/hotels.csv")
    
    X = df.drop(['price'], axis=1)
    y = df['price']
    
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, "hotel_price_model.pkl")
    print("✅ Hotel price model saved")

def retrain_flight_models():
    """Retrain and save flight cancellation and price models"""
    print("Training flight models...")
    df = pd.read_csv("data/flights.csv")
    
    # Flight cancellation model
    X_cancel = df[['Distance_Vol_KM', 'Nombre_Escales', 'Saison_Touristique']]
    y_cancel = df['is_cancelled']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cancel)
    
    model_rf = DecisionTreeClassifier(random_state=42)
    model_rf.fit(X_scaled, y_cancel)
    
    joblib.dump(model_rf, "model_rf.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Flight cancellation model saved")
    
    # Flight price model
    X_price = df[['Nombre_Escales', 'Taxe_Price', 'AirlineName', 'Region', 'Mois', 'Jour']]
    y_price = df['price']
    
    model_flight = DecisionTreeRegressor(random_state=42)
    model_flight.fit(X_price, y_price)
    
    joblib.dump(model_flight, "model_flight.pkl")
    print("✅ Flight price model saved")

if __name__ == "__main__":
    print("🔄 Starting model retraining...")
    retrain_hotel_model()
    retrain_flight_models()
    print("✅ All models retrained successfully!")
