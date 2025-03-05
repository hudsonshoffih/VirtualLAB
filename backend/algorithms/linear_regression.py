import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_housing_data():
    """
    Load the housing dataset for linear regression practice
    """
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
    return pd.read_csv(url)

def prepare_data(df, features=None, target='price'):
    """
    Prepare data for linear regression
    """
    if features is None:
        features = ['area', 'bedrooms', 'bathrooms', 'stories']
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, features

def train_model(X_train, y_train):
    """
    Train a linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the linear regression model
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    
    return mse, r2

def interpret_coefficients(model, features):
    """
    Interpret the coefficients of the linear regression model
    """
    coefficients = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    })
    
    print("Model Coefficients:")
    print(coefficients)
    print(f"Intercept: {model.intercept_:.2f}")
    
    # Plot coefficients
    plt.figure(figsize=(10, 6))
    plt.barh(coefficients['Feature'], coefficients['Coefficient'])
    plt.xlabel('Coefficient Value')
    plt.title('Feature Coefficients')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    return coefficients

