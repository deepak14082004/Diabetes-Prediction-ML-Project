import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data():
    """
    Load the diabetes dataset from sklearn
    Returns preprocessed features and target
    """
    # Using the Pima Indians Diabetes Dataset
    df = pd.read_excel('diabetes.xls')
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    return X, y

def preprocess_data(X, y):
    """
    Preprocess the data by splitting and scaling
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Train a Random Forest model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))

def predict_diabetes(model, scaler, patient_data):
    """
    Make prediction for new patient data
    """
    # Scale the input data
    scaled_data = scaler.transform([patient_data])
    # Make prediction
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)
    return prediction[0], probability[0]

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test_scaled, y_test)
    
    # Save model and scaler
    joblib.dump(model, 'diabetes_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("\nModel and scaler saved successfully!")
    
    # Example prediction
    print("\nExample prediction with sample data:")
    sample_patient = [6,148,72,35,0,33.6,0.627,50 ]  # Sample patient data
    prediction, probability = predict_diabetes(model, scaler, sample_patient)
    print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
    print(f"Probability: {probability[1]:.2f}")

if __name__ == "__main__":
    main()
