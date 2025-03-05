import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_iris_data():
    """
    Load the Iris dataset for SVM practice
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    return X, y, feature_names, target_names

def visualize_iris_data(X, y, feature_names, target_names):
    """
    Visualize the Iris dataset
    """
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = df['target'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})
    
    # Plot sepal length vs sepal width
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[feature_names[0]], y=df[feature_names[1]], hue=df['species'], palette='viridis')
    plt.title('Iris Dataset: Sepal Length vs Sepal Width')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    
    return df

def train_svm_model(X, y, kernel='linear'):
    """
    Train an SVM model on the Iris dataset
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the SVM
    svm = SVC(kernel=kernel)
    svm.fit(X_train_scaled, y_train)
    
    return svm, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_svm_model(svm, X_test_scaled, y_test, target_names):
    """
    Evaluate the SVM model
    """
    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    return accuracy, y_pred

def visualize_decision_boundaries(X, y, svm, scaler, target_names):
    """
    Visualize the decision boundaries of the SVM model
    """
    # We'll use only the first two features for visualization
    X_2d = X[:, :2]
    X_2d_scaled = scaler.transform(X_2d)
    
    # Train a new SVM on just these two features
    svm_2d = SVC(kernel='linear')
    svm_2d.fit(X_2d_scaled, y)
    
    # Create a mesh grid
    x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
    y_min, y_max = X_2d_scaled[:, 1].min() - 1, X_2d_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict class for each point in the mesh
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and the data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot the data points
    for i, color in enumerate(['navy', 'turquoise', 'darkorange']):
        idx = np.where(y == i)
        plt.scatter(X_2d_scaled[idx, 0], X_2d_scaled[idx, 1], c=color, label=target_names[i],
                   edgecolor='black', s=40)
    
    plt.xlabel('Scaled Sepal Length')
    plt.ylabel('Scaled Sepal Width')
    plt.title('SVM Decision Boundaries (First Two Features)')
    plt.legend()
    
    return svm_2d

