"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Info, HelpCircle, BarChart4, LineChart, ScatterChart, PieChart } from 'lucide-react'
import {
  LineChart as ReLineChart,
  Line,
  BarChart as ReBarChart,
  Bar,
  ScatterChart as ReScatterChart,
  Scatter,
  PieChart as RePieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ZAxis
} from 'recharts'
import type { Algorithm } from "@/lib/types"
import { 
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { cn } from "@/lib/utils"

interface DemoTabProps {
  algorithm: Algorithm
}

// Define dataset options
const datasets = [
  { id: "housing", name: "Housing Prices", description: "Boston housing price dataset with 13 features" },
  { id: "iris", name: "Iris Flowers", description: "Classic iris flower dataset with 3 classes" },
  { id: "diabetes", name: "Diabetes", description: "Diabetes progression dataset with 10 features" },
  { id: "wine", name: "Wine Quality", description: "Wine quality dataset with chemical properties" },
  { id: "breast_cancer", name: "Breast Cancer", description: "Breast cancer diagnostic dataset" },
]

// Algorithm-specific controls and explanations
const algorithmConfigs: Record<string, {
  controls: Array<{
    name: string,
    label: string,
    min: number,
    max: number,
    step: number,
    defaultValue: number,
    description: string
  }>,
  metrics: Array<{
    name: string,
    label: string,
    description: string
  }>,
  explanation: string,
  code: string,
  chartType: "line" | "bar" | "scatter" | "pie" | "heatmap" | "multiple"
}> = {
  "eda": {
    controls: [
      {
        name: "binSize",
        label: "Bin Size",
        min: 5,
        max: 50,
        step: 5,
        defaultValue: 20,
        description: "Number of bins used in histograms for continuous variables"
      },
      {
        name: "outlierThreshold",
        label: "Outlier Threshold",
        min: 1.5,
        max: 3.5,
        step: 0.1,
        defaultValue: 2.0,
        description: "IQR multiplier for outlier detection (higher = fewer outliers)"
      }
    ],
    metrics: [
      {
        name: "missingValues",
        label: "Missing Values",
        description: "Percentage of missing values in the dataset"
      },
      {
        name: "outliers",
        label: "Outliers",
        description: "Percentage of data points identified as outliers"
      },
      {
        name: "skewness",
        label: "Skewness",
        description: "Measure of asymmetry in the data distribution"
      },
      {
        name: "kurtosis",
        label: "Kurtosis",
        description: "Measure of 'tailedness' of the data distribution"
      }
    ],
    explanation: "Exploratory Data Analysis (EDA) is the process of analyzing and visualizing datasets to summarize their main characteristics. It helps identify patterns, spot anomalies, test hypotheses, and check assumptions through statistical graphics and other data visualization methods.",
    code: `# Python code for basic EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('dataset.csv')

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='feature', bins=20)
plt.title('Distribution of Feature')
plt.show()

# Detect outliers using IQR method
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['feature'] < Q1 - 1.5 * IQR) | 
              (df['feature'] > Q3 + 1.5 * IQR)]`,
    chartType: "multiple"
  },
  "regression": {
    controls: [
      {
        name: "weight",
        label: "Weight",
        min: -4,
        max: 4,
        step: 0.01,
        defaultValue: -0.88,
        description: "Coefficient/slope of the linear model"
      },
      {
        name: "bias",
        label: "Bias",
        min: 10,
        max: 24,
        step: 0.1,
        defaultValue: 17,
        description: "Y-intercept of the linear model"
      },
      {
        name: "noiseLevel",
        label: "Noise Level",
        min: 0,
        max: 5,
        step: 0.1,
        defaultValue: 1,
        description: "Amount of random noise added to the data"
      }
    ],
    metrics: [
      {
        name: "r2",
        label: "R²",
        description: "Coefficient of determination - proportion of variance explained by the model"
      },
      {
        name: "mse",
        label: "MSE",
        description: "Mean Squared Error - average squared difference between predicted and actual values"
      },
      {
        name: "rmse",
        label: "RMSE",
        description: "Root Mean Squared Error - square root of MSE, in same units as target variable"
      },
      {
        name: "mae",
        label: "MAE",
        description: "Mean Absolute Error - average absolute difference between predicted and actual values"
      }
    ],
    explanation: "Linear Regression is a statistical method that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. The model assumes a linear relationship between variables, where the dependent variable can be calculated as a linear combination of the independent variables.",
    code: `# Python code for Linear Regression
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 6])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Get model parameters
weight = model.coef_[0]
bias = model.intercept_

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
r2 = model.score(X, y)

# Plot results
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()`,
    chartType: "line"
  },
  "classification": {
    controls: [
      {
        name: "decisionBoundary",
        label: "Decision Boundary",
        min: -3,
        max: 3,
        step: 0.1,
        defaultValue: 0,
        description: "Threshold for class separation"
      },
      {
        name: "featureWeight1",
        label: "Feature 1 Weight",
        min: -2,
        max: 2,
        step: 0.1,
        defaultValue: 1,
        description: "Importance of feature 1 in classification"
      },
      {
        name: "featureWeight2",
        label: "Feature 2 Weight",
        min: -2,
        max: 2,
        step: 0.1,
        defaultValue: 1,
        description: "Importance of feature 2 in classification"
      }
    ],
    metrics: [
      {
        name: "accuracy",
        label: "Accuracy",
        description: "Proportion of correctly classified instances"
      },
      {
        name: "precision",
        label: "Precision",
        description: "Proportion of positive identifications that were actually correct"
      },
      {
        name: "recall",
        label: "Recall",
        description: "Proportion of actual positives that were correctly identified"
      },
      {
        name: "f1",
        label: "F1 Score",
        description: "Harmonic mean of precision and recall"
      }
    ],
    explanation: "Logistic Regression is a statistical method for binary classification that estimates the probability of a binary outcome based on one or more predictor variables. It uses a logistic function to model a binary dependent variable, making it suitable for classification problems where the outcome is categorical.",
    code: `# Python code for Logistic Regression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Generate sample data for binary classification
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Get model parameters
weights = model.coef_[0]
bias = model.intercept_[0]

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')

# Create a grid to evaluate model
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()`,
    chartType: "scatter"
  },
  "clustering": {
    controls: [
      {
        name: "clusters",
        label: "Number of Clusters",
        min: 2,
        max: 10,
        step: 1,
        defaultValue: 3,
        description: "Number of clusters to identify in the data"
      },
      {
        name: "maxIterations",
        label: "Max Iterations",
        min: 10,
        max: 100,
        step: 10,
        defaultValue: 50,
        description: "Maximum number of iterations for convergence"
      },
      {
        name: "randomSeed",
        label: "Random Seed",
        min: 1,
        max: 100,
        step: 1,
        defaultValue: 42,
        description: "Seed for random initialization of cluster centers"
      }
    ],
    metrics: [
      {
        name: "inertia",
        label: "Inertia",
        description: "Sum of squared distances to the nearest cluster center"
      },
      {
        name: "silhouette",
        label: "Silhouette Score",
        description: "Measure of how similar an object is to its own cluster compared to other clusters"
      },
      {
        name: "calinski",
        label: "Calinski-Harabasz Index",
        description: "Ratio of between-cluster variance to within-cluster variance"
      },
      {
        name: "davies",
        label: "Davies-Bouldin Index",
        description: "Average similarity between clusters (lower is better)"
      }
    ],
    explanation: "Clustering is an unsupervised learning technique that groups similar data points together based on certain features. K-means clustering partitions data into k clusters where each observation belongs to the cluster with the nearest mean. It's widely used for customer segmentation, image compression, and anomaly detection.",
    code: `# Python code for K-means clustering
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate sample data
X = np.random.rand(100, 2) * 10

# Create and fit the model
kmeans = KMeans(n_clusters=3, random_state=42, max_iter=50)
kmeans.fit(X)

# Get cluster assignments and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Calculate metrics
inertia = kmeans.inertia_
silhouette = silhouette_score(X, labels)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()`,
    chartType: "scatter"
  },
  "dimensionality": {
    controls: [
      {
        name: "components",
        label: "Number of Components",
        min: 2,
        max: 10,
        step: 1,
        defaultValue: 2,
        description: "Number of principal components to retain"
      },
      {
        name: "perplexity",
        label: "Perplexity (t-SNE)",
        min: 5,
        max: 50,
        step: 5,
        defaultValue: 30,
        description: "Balance between local and global structure in t-SNE"
      },
      {
        name: "learningRate",
        label: "Learning Rate",
        min: 10,
        max: 1000,
        step: 10,
        defaultValue: 200,
        description: "Learning rate for t-SNE optimization"
      }
    ],
    metrics: [
      {
        name: "explainedVariance",
        label: "Explained Variance",
        description: "Percentage of variance explained by the principal components"
      },
      {
        name: "kl_divergence",
        label: "KL Divergence",
        description: "Kullback-Leibler divergence between high and low-dimensional distributions (t-SNE)"
      }
    ],
    explanation: "Dimensionality Reduction techniques transform high-dimensional data into a lower-dimensional space while preserving important information. Principal Component Analysis (PCA) finds orthogonal axes that maximize variance, while t-SNE (t-Distributed Stochastic Neighbor Embedding) preserves local relationships, making it effective for visualization.",
    code: `# Python code for PCA and t-SNE
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load sample data
digits = load_digits()
X = digits.data
y = digits.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_.sum()

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PCA plot
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
ax1.set_title(f'PCA (Explained Variance: {explained_variance:.2f})')
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')

# t-SNE plot
scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
ax2.set_title('t-SNE')
ax2.set_xlabel('t-SNE Component 1')
ax2.set_ylabel('t-SNE Component 2')

plt.colorbar(scatter1, ax=ax1, label='Digit')
plt.colorbar(scatter2, ax=ax2, label='Digit')
plt.tight_layout()
plt.show()`,
    chartType: "scatter"
  },
  "neural_networks": {
    controls: [
      {
        name: "learningRate",
        label: "Learning Rate",
        min: 0.001,
        max: 0.1,
        step: 0.001,
        defaultValue: 0.01,
        description: "Step size for gradient descent optimization"
      },
      {
        name: "hiddenLayers",
        label: "Hidden Layers",
        min: 1,
        max: 5,
        step: 1,
        defaultValue: 2,
        description: "Number of hidden layers in the network"
      },
      {
        name: "neuronsPerLayer",
        label: "Neurons Per Layer",
        min: 2,
        max: 32,
        step: 2,
        defaultValue: 16,
        description: "Number of neurons in each hidden layer"
      },
      {
        name: "epochs",
        label: "Epochs",
        min: 10,
        max: 200,
        step: 10,
        defaultValue: 100,
        description: "Number of complete passes through the training dataset"
      }
    ],
    metrics: [
      {
        name: "accuracy",
        label: "Accuracy",
        description: "Proportion of correctly classified instances"
      },
      {
        name: "loss",
        label: "Loss",
        description: "Value of the loss function (e.g., cross-entropy)"
      },
      {
        name: "val_accuracy",
        label: "Validation Accuracy",
        description: "Accuracy on validation data"
      },
      {
        name: "val_loss",
        label: "Validation Loss",
        description: "Loss on validation data"
      }
    ],
    explanation: "Neural Networks are computational models inspired by the human brain's structure and function. They consist of interconnected nodes (neurons) organized in layers that process information. Deep learning uses neural networks with many layers to learn hierarchical representations of data, enabling them to solve complex tasks like image recognition and natural language processing.",
    code: `# Python code for a simple neural network using TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X_train = np.random.rand(1000, 10)
y_train = (X_train.sum(axis=1) > 5).astype(int)
X_val = np.random.rand(200, 10)
y_val = (X_val.sum(axis=1) > 5).astype(int)

# Create model
model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()`,
    chartType: "line"
  },
  "nlp": {
    controls: [
      {
        name: "vocabSize",
        label: "Vocabulary Size",
        min: 1000,
        max: 10000,
        step: 1000,
        defaultValue: 5000,
        description: "Maximum number of words in the vocabulary"
      },
      {
        name: "maxSequenceLength",
        label: "Max Sequence Length",
        min: 10,
        max: 100,
        step: 10,
        defaultValue: 50,
        description: "Maximum length of text sequences"
      },
      {
        name: "embeddingDim",
        label: "Embedding Dimension",
        min: 32,
        max: 256,
        step: 32,
        defaultValue: 128,
        description: "Dimension of word embeddings"
      }
    ],
    metrics: [
      {
        name: "accuracy",
        label: "Accuracy",
        description: "Proportion of correctly classified texts"
      },
      {
        name: "precision",
        label: "Precision",
        description: "Proportion of positive identifications that were actually correct"
      },
      {
        name: "recall",
        label: "Recall",
        description: "Proportion of actual positives that were correctly identified"
      },
      {
        name: "f1",
        label: "F1 Score",
        description: "Harmonic mean of precision and recall"
      }
    ],
    explanation: "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves techniques for analyzing, understanding, and generating human language. Common NLP tasks include sentiment analysis, text classification, machine translation, and question answering.",
    code: `# Python code for text classification with TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Sample text data
texts = [
    "I love this movie, it's amazing",
    "This movie is terrible, I hated it",
    "Great film, highly recommended",
    "Worst movie ever, don't watch it",
    # ... more examples
]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Tokenize text
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_length = 50
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Create model
embedding_dim = 128
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    padded_sequences, np.array(labels),
    epochs=20,
    validation_split=0.2
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()`,
    chartType: "bar"
  },
  "time_series": {
    controls: [
      {
        name: "windowSize",
        label: "Window Size",
        min: 3,
        max: 30,
        step: 1,
        defaultValue: 10,
        description: "Number of time steps to use for prediction"
      },
      {
        name: "forecastHorizon",
        label: "Forecast Horizon",
        min: 1,
        max: 20,
        step: 1,
        defaultValue: 5,
        description: "Number of time steps to predict into the future"
      },
      {
        name: "seasonality",
        label: "Seasonality Period",
        min: 0,
        max: 12,
        step: 1,
        defaultValue: 4,
        description: "Length of seasonal patterns (0 for no seasonality)"
      }
    ],
    metrics: [
      {
        name: "mse",
        label: "MSE",
        description: "Mean Squared Error between predicted and actual values"
      },
      {
        name: "rmse",
        label: "RMSE",
        description: "Root Mean Squared Error"
      },
      {
        name: "mae",
        label: "MAE",
        description: "Mean Absolute Error"
      },
      {
        name: "mape",
        label: "MAPE",
        description: "Mean Absolute Percentage Error"
      }
    ],
    explanation: "Time Series Analysis involves analyzing data points collected or indexed in time order to extract meaningful statistics and characteristics. It's used for forecasting future values, understanding patterns, and identifying trends and seasonal variations in data collected over time.",
    code: `# Python code for time series forecasting with ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
trend = np.linspace(0, 5, 100)
seasonality = 2 * np.sin(np.linspace(0, 4*np.pi, 100))
noise = np.random.normal(0, 0.5, 100)
y = trend + seasonality + noise

# Create time series
ts = pd.Series(y, index=dates)

# Split into train and test
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Calculate metrics
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / test)) * 100

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Values')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.fill_between(test.index, 
                 forecast - 1.96 * np.std(model_fit.resid),
                 forecast + 1.96 * np.std(model_fit.resid),
                 color='red', alpha=0.2)
plt.title('ARIMA Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()`,
    chartType: "line"
  },
  "reinforcement": {
    controls: [
      {
        name: "learningRate",
        label: "Learning Rate",
        min: 0.01,
        max: 1,
        step: 0.01,
        defaultValue: 0.1,
        description: "Rate at which the agent learns from new information"
      },
      {
        name: "discountFactor",
        label: "Discount Factor",
        min: 0.5,
        max: 0.99,
        step: 0.01,
        defaultValue: 0.9,
        description: "Importance of future rewards compared to immediate rewards"
      },
      {
        name: "explorationRate",
        label: "Exploration Rate",
        min: 0.01,
        max: 0.5,
        step: 0.01,
        defaultValue: 0.2,
        description: "Probability of taking a random action (exploration)"
      }
    ],
    metrics: [
      {
        name: "cumulativeReward",
        label: "Cumulative Reward",
        description: "Total reward accumulated over time"
      },
      {
        name: "averageReward",
        label: "Average Reward",
        description: "Average reward per episode"
      },
      {
        name: "episodesCompleted",
        label: "Episodes Completed",
        description: "Number of completed episodes"
      },
      {
        name: "convergenceTime",
        label: "Convergence Time",
        description: "Number of episodes until policy stabilizes"
      }
    ],
    explanation: "Reinforcement Learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. The agent learns through trial and error, receiving feedback in the form of rewards or penalties. It's used in robotics, game playing, an  receiving feedback in the form of rewards or penalties. It's used in robotics, game playing, and autonomous systems like self-driving cars.",
    code: `# Python code for Q-learning (a simple reinforcement learning algorithm)
import numpy as np
import matplotlib.pyplot as plt

# Define a simple grid world environment
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = [0, 0]
        self.goal_pos = [size-1, size-1]
        self.reset()
        
    def reset(self):
        self.agent_pos = [0, 0]
        return self.agent_pos.copy()
        
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.size - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.size - 1:
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
            
        # Check if goal reached
        done = (self.agent_pos == self.goal_pos)
        reward = 1 if done else -0.01  # Small penalty for each step
        
        return self.agent_pos.copy(), reward, done

# Q-learning algorithm
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.2):
    # Initialize Q-table
    q_table = np.zeros((env.size, env.size, 4))
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(4)  # Random action
            else:
                action = np.argmax(q_table[state[0], state[1]])  # Best action
                
            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Update Q-value
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1]])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state[0], state[1], action] = new_value
            
            state = next_state
            
        rewards_history.append(total_reward)
        
    return q_table, rewards_history

# Run algorithm
env = GridWorld(size=5)
q_table, rewards = q_learning(env, episodes=500)

# Plot rewards over time
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()`,
    chartType: "line"
  },
  "computer_vision": {
    controls: [
      {
        name: "kernelSize",
        label: "Kernel Size",
        min: 3,
        max: 11,
        step: 2,
        defaultValue: 5,
        description: "Size of convolutional kernel"
      },
      {
        name: "filters",
        label: "Number of Filters",
        min: 8,
        max: 64,
        step: 8,
        defaultValue: 32,
        description: "Number of convolutional filters"
      },
      {
        name: "learningRate",
        label: "Learning Rate",
        min: 0.0001,
        max: 0.01,
        step: 0.0001,
        defaultValue: 0.001,
        description: "Learning rate for model training"
      }
    ],
    metrics: [
      {
        name: "accuracy",
        label: "Accuracy",
        description: "Proportion of correctly classified images"
      },
      {
        name: "precision",
        label: "Precision",
        description: "Proportion of positive identifications that were actually correct"
      },
      {
        name: "recall",
        label: "Recall",
        description: "Proportion of actual positives that were correctly identified"
      },
      {
        name: "f1",
        label: "F1 Score",
        description: "Harmonic mean of precision and recall"
      }
    ],
    explanation: "Computer Vision is a field of AI that enables computers to interpret and understand visual information from the world. It involves techniques for acquiring, processing, analyzing, and understanding digital images to produce numerical or symbolic information. Applications include image classification, object detection, facial recognition, and autonomous vehicles.",
    code: `# Python code for image classification with CNN using TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data (using MNIST as example)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Create CNN model
model = Sequential([
    Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Visualize predictions
predictions = model.predict(x_test[:10])
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i in range(10):
    axes[i].imshow(x_test[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f'Pred: {np.argmax(predictions[i])}\\nTrue: {y_test[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()`,
    chartType: "multiple"
  },
  "ensemble": {
    controls: [
      {
        name: "numEstimators",
        label: "Number of Estimators",
        min: 10,
        max: 200,
        step: 10,
        defaultValue: 100,
        description: "Number of base models in the ensemble"
      },
      {
        name: "maxDepth",
        label: "Max Depth",
        min: 3,
        max: 15,
        step: 1,
        defaultValue: 8,
        description: "Maximum depth of decision trees"
      },
      {
        name: "minSamplesSplit",
        label: "Min Samples Split",
        min: 2,
        max: 20,
        step: 1,
        defaultValue: 5,
        description: "Minimum samples required to split a node"
      }
    ],
    metrics: [
      {
        name: "accuracy",
        label: "Accuracy",
        description: "Proportion of correctly classified instances"
      },
      {
        name: "precision",
        label: "Precision",
        description: "Proportion of positive identifications that were actually correct"
      },
      {
        name: "recall",
        label: "Recall",
        description: "Proportion of actual positives that were correctly identified"
      },
      {
        name: "f1",
        label: "F1 Score",
        description: "Harmonic mean of precision and recall"
      }
    ],
    explanation: "Ensemble Methods combine multiple machine learning models to improve performance beyond what any individual model could achieve. Random Forest combines many decision trees, each trained on a random subset of data and features. Boosting methods like AdaBoost and Gradient Boosting train models sequentially, with each new model focusing on the errors of previous ones.",
    code: `# Python code for Random Forest and Gradient Boosting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate synthetic data
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, 
    n_classes=2, random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100, max_depth=8, min_samples_split=5, random_state=42
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Train Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100, max_depth=8, min_samples_split=5, random_state=42
)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)

# Calculate metrics
models = {
    'Random Forest': rf_preds,
    'Gradient Boosting': gb_preds
}

metrics = {}
for name, preds in models.items():
    metrics[name] = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds)
    }

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(10), rf.feature_importances_[:10])
plt.title('Random Forest Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')

plt.subplot(1, 2, 2)
plt.bar(range(10), gb.feature_importances_[:10])
plt.title('Gradient Boosting Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')

plt.tight_layout()
plt.show()

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_preds, ax=axes[0], cmap='Blues'
)
axes[0].set_title('Random Forest Confusion Matrix')

ConfusionMatrixDisplay.from_predictions(
    y_test, gb_preds, ax=axes[1], cmap='Blues'
)
axes[1].set_title('Gradient Boosting Confusion Matrix')

plt.tight_layout()
plt.show()`,
    chartType: "bar"
  }
}

// Generate sample data based on algorithm type
const generateData = (algorithmId: string, params: Record<string, number>) => {
  switch (algorithmId) {
    case "regression":
      const { weight, bias, noiseLevel } = params;
      const regressionData = [];
      for (let x = 0; x <= 4; x += 0.2) {
        const y = weight * x + bias + (Math.random() * 2 - 1) * noiseLevel;
        regressionData.push({ x, y });
      }
      return regressionData;
      
    case "classification":
      const { decisionBoundary, featureWeight1, featureWeight2 } = params;
      const classificationData = [];
      for (let i = 0; i < 100; i++) {
        const x = Math.random() * 10 - 5;
        const y = Math.random() * 10 - 5;
        const value = featureWeight1 * x + featureWeight2 * y;
        const category = value > decisionBoundary ? "A" : "B";
        classificationData.push({ x, y, category });
      }
      return classificationData;
      
    case "clustering":
      const { clusters } = params;
      const clusteringData: { x: number; y: number; cluster: string }[] = [];
      // Generate cluster centers
      const centers = Array.from({ length: clusters }, () => ({
        x: Math.random() * 8 - 4,
        y: Math.random() * 8 - 4
      }));
      
      // Generate points around centers
      centers.forEach((center, clusterIdx) => {
        for (let i = 0; i < 20; i++) {
          clusteringData.push({
            x: center.x + Math.random() * 2 - 1,
            y: center.y + Math.random() * 2 - 1,
            cluster: `Cluster ${clusterIdx + 1}`
          });
        }
      });
      return clusteringData;
      
    case "neural_networks":
      const { epochs } = params;
      const nnData = [];
      for (let i = 0; i < epochs; i++) {
        nnData.push({
          epoch: i + 1,
          accuracy: Math.min(0.5 + i / epochs * 0.45 + Math.random() * 0.05, 1),
          loss: Math.max(0.5 - i / epochs * 0.45 + Math.random() * 0.05, 0.05)
        });
      }
      return nnData;
      
    case "time_series":
      const { seasonality } = params;
      const tsData = [];
      for (let i = 0; i < 100; i++) {
        const trend = i * 0.05;
        const seasonal = seasonality > 0 ? Math.sin(i * 2 * Math.PI / seasonality) * 2 : 0;
        const noise = Math.random() * 0.5 - 0.25;
        tsData.push({
          time: i,
          value: trend + seasonal + noise
        });
      }
      return tsData;
      
    case "reinforcement":
      const { explorationRate } = params;
      const rlData = [];
      let reward = 0;
      for (let i = 0; i < 100; i++) {
        // Simulate learning curve with exploration rate impact
        const improvement = Math.log(i + 1) * (1 - explorationRate * 0.5);
        const noise = (Math.random() - 0.5) * (1 - i / 100);
        reward = Math.min(Math.max(reward + improvement * 0.1 + noise, 0), 10);
        rlData.push({
          episode: i + 1,
          reward
        });
      }
      return rlData;
      
    case "eda":
      // Generate histogram data
      const histogramData = [];
      for (let i = 0; i < 10; i++) {
        histogramData.push({
          bin: i,
          frequency: Math.floor(Math.random() * 50) + 10
        });
      }
      return histogramData;
      
    default:
      // Generic data for other algorithms
      const genericData = [];
      for (let i = 0; i < 20; i++) {
        genericData.push({
          x: i,
          y: Math.random() * 10
        });
      }
      return genericData;
  }
};

// Generate metrics based on algorithm type and parameters
const generateMetrics = (algorithmId: string, params: Record<string, number>) => {
  const baseMetrics: Record<string, number> = {};
  
  const config = algorithmConfigs[algorithmId];
  if (!config) return baseMetrics;
  
  config.metrics.forEach(metric => {
    // Generate realistic-looking metrics based on parameters
    let value = 0;
    
    switch (metric.name) {
      case "r2":
        value = Math.random() * 0.3 + 0.7; // 0.7-1.0
        break;
      case "accuracy":
        value = Math.random() * 0.2 + 0.8; // 0.8-1.0
        break;
      case "precision":
      case "recall":
      case "f1":
        value = Math.random() * 0.3 + 0.7; // 0.7-1.0
        break;
      case "mse":
        value = Math.random() * 10; // 0-10
        break;
      case "rmse":
        value = Math.random() * 3; // 0-3
        break;
      case "mae":
        value = Math.random() * 2; // 0-2
        break;
      case "silhouette":
        value = Math.random() * 0.6 + 0.4; // 0.4-1.0
        break;
      default:
        value = Math.random() * 100; // 0-100
    }
    
    baseMetrics[metric.name] = parseFloat(value.toFixed(2));
  });
  
  return baseMetrics;
};

export function DemoTab({ algorithm }: DemoTabProps) {
  const config = algorithmConfigs[algorithm.id] || algorithmConfigs["regression"];
  
  // Initialize state for controls
  const [params, setParams] = useState<Record<string, number>>({});
  const [selectedDataset, setSelectedDataset] = useState(datasets[0]);
  const [data, setData] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<Record<string, number>>({});
  
  // Initialize parameters on algorithm change
  useEffect(() => {
    const initialParams: Record<string, number> = {};
    config.controls.forEach(control => {
      initialParams[control.name] = control.defaultValue;
    });
    setParams(initialParams);
    
    // Reset dataset when algorithm changes
    setSelectedDataset(datasets[0]);
  }, [algorithm.id]);
  
  // Update data and metrics when parameters or dataset changes
  useEffect(() => {
    if (Object.keys(params).length === 0) return;
    
    const newData = generateData(algorithm.id, params);
    setData(newData);
    
    const newMetrics = generateMetrics(algorithm.id, params);
    setMetrics(newMetrics);
  }, [params, selectedDataset, algorithm.id]);
  
  // Handle parameter change
  const handleParamChange = (name: string, value: number[]) => {
    setParams(prev => ({
      ...prev,
      [name]: value[0]
    }));
  };
  
  // Render appropriate chart based on algorithm type
  const renderChart = () => {
    if (!data || data.length === 0) return null;
    
    switch (config.chartType) {
      case "line":
        return (
          <ResponsiveContainer width="100%" height="100%">
            <ReLineChart
              data={data}
              margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey={data[0].hasOwnProperty('x') ? 'x' : 
                        data[0].hasOwnProperty('time') ? 'time' : 
                        data[0].hasOwnProperty('epoch') ? 'epoch' : 
                        'episode'}
              />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey={data[0].hasOwnProperty('y') ? 'y' : 
                         data[0].hasOwnProperty('value') ? 'value' : 
                         data[0].hasOwnProperty('accuracy') ? 'accuracy' : 
                         'reward'}
                stroke="#8884d8" 
                activeDot={{ r: 8 }}
              />
              {data[0].hasOwnProperty('loss') && (
                <Line type="monotone" dataKey="loss" stroke="#82ca9d" />
              )}
            </ReLineChart>
          </ResponsiveContainer>
        );
        
      case "bar":
        return (
          <ResponsiveContainer width="100%" height="100%">
            <ReBarChart
              data={data}
              margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey={data[0].hasOwnProperty('bin') ? 'bin' : 
                        data[0].hasOwnProperty('x') ? 'x' : 
                        'category'}
              />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar 
                dataKey={data[0].hasOwnProperty('frequency') ? 'frequency' : 
                         data[0].hasOwnProperty('y') ? 'y' : 
                         'value'}
                fill="#8884d8" 
              />
            </ReBarChart>
          </ResponsiveContainer>
        );
        
      case "scatter":
        return (
          <ResponsiveContainer width="100%" height="100%">
            <ReScatterChart
              margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
            >
              <CartesianGrid />
              <XAxis type="number" dataKey="x" name="X" />
              <YAxis type="number" dataKey="y" name="Y" />
              <ZAxis range={[60, 60]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              <Scatter 
                name={data[0].hasOwnProperty('category') ? 'Category A' : 
                      data[0].hasOwnProperty('cluster') ? 'Clusters' : 
                      'Data Points'}
                data={data.filter((d: any) => !d.category || d.category === 'A')}
                fill="#8884d8"
              />
              {data[0].hasOwnProperty('category') && (
                <Scatter 
                  name="Category B" 
                  data={data.filter((d: any) => d.category === 'B')}
                  fill="#82ca9d"
                />
              )}
            </ReScatterChart>
          </ResponsiveContainer>
        );
        
      case "pie":
        return (
          <ResponsiveContainer width="100%" height="100%">
            <RePieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={`#${Math.floor(Math.random() * 16777215).toString(16)}`} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </RePieChart>
          </ResponsiveContainer>
        );
        
      case "multiple":
        // For EDA, show multiple chart types
        if (algorithm.id === "eda") {
          return (
            <div className="grid grid-cols-2 gap-4 h-full">
              <div className="border rounded-lg p-4">
                <h3 className="text-sm font-medium mb-2">Distribution</h3>
                <div className="h-[calc(100%-2rem)]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ReBarChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="bin" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="frequency" fill="#8884d8" />
                    </ReBarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              <div className="border rounded-lg p-4">
                <h3 className="text-sm font-medium mb-2">Correlation</h3>
                <div className="h-[calc(100%-2rem)]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ReScatterChart margin={{ top: 20, right: 20, left: 20, bottom: 20 }}>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="bin" name="Feature 1" />
                      <YAxis type="number" dataKey="frequency" name="Feature 2" />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter name="Data Points" data={data} fill="#8884d8" />
                    </ReScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          );
        }
        
        // For computer vision, show multiple visualizations
        if (algorithm.id === "computer_vision") {
          return (
            <div className="grid grid-cols-2 gap-4 h-full">
              <div className="border rounded-lg p-4">
                <h3 className="text-sm font-medium mb-2">Training Progress</h3>
                <div className="h-[calc(100%-2rem)]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ReLineChart data={Array.from({ length: 10 }, (_, i) => ({ epoch: i + 1, accuracy: 0.5 + i * 0.05, loss: 0.5 - i * 0.04 }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="accuracy" stroke="#8884d8" />
                      <Line type="monotone" dataKey="loss" stroke="#82ca9d" />
                    </ReLineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              <div className="border rounded-lg p-4">
                <h3 className="text-sm font-medium mb-2">Confusion Matrix</h3>
                <div className="h-[calc(100%-2rem)] flex items-center justify-center">
                  <div className="grid grid-cols-3 grid-rows-3 gap-1">
                    <div className="bg-muted p-2 text-center font-medium">Predicted</div>
                    <div className="bg-muted p-2 text-center">Class A</div>
                    <div className="bg-muted p-2 text-center">Class B</div>
                    <div className="bg-muted p-2 text-center font-medium">Actual</div>
                    <div className="bg-green-100 dark:bg-green-900/30 p-2 text-center font-medium">85%</div>
                    <div className="bg-red-100 dark:bg-red-900/30 p-2 text-center">15%</div>
                    <div className="bg-muted p-2 text-center">Class B</div>
                    <div className="bg-red-100 dark:bg-red-900/30 p-2 text-center">12%</div>
                    <div className="bg-green-100 dark:bg-green-900/30 p-2 text-center font-medium">88%</div>
                  </div>
                </div>
              </div>
            </div>
          );
        }
        
        return (
          <ResponsiveContainer width="100%" height="100%">
            <ReLineChart
              data={data}
              margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="y" stroke="#8884d8" />
            </ReLineChart>
          </ResponsiveContainer>
        );
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2">
        <h2 className="text-2xl font-bold">{algorithm.title} Demo</h2>
        <p className="text-muted-foreground">
          Interact with the parameters to see how they affect the algorithm's behavior.
        </p>
      </div>

      <div className="flex justify-between items-center">
        <Select
          value={selectedDataset.id}
          onValueChange={(value) => {
            const dataset = datasets.find(d => d.id === value) || datasets[0];
            setSelectedDataset(dataset);
          }}
        >
          <SelectTrigger className="w-[250px]">
            <SelectValue placeholder="Select dataset" />
          </SelectTrigger>
          <SelectContent>
            {datasets.map((dataset) => (
              <SelectItem key={dataset.id} value={dataset.id}>
                {dataset.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="font-mono">
            {selectedDataset.name}
          </Badge>
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <Info className="h-4 w-4" />
                <span className="sr-only">Dataset info</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80">
              <div className="space-y-2">
                <h4 className="font-medium">{selectedDataset.name}</h4>
                <p className="text-sm text-muted-foreground">{selectedDataset.description}</p>
                <div className="text-xs text-muted-foreground">
                  <span className="font-medium">Features:</span> {selectedDataset.id === 'housing' ? '13' : selectedDataset.id === 'iris' ? '4' : '10'} | 
                  <span className="font-medium"> Samples:</span> {selectedDataset.id === 'housing' ? '506' : selectedDataset.id === 'iris' ? '150' : '442'}
                </div>
              </div>
            </PopoverContent>
          </Popover>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-[2fr,1fr]">
        <Card className="p-6">
          <div className="h-[400px] w-full">
            {renderChart()}
          </div>
        </Card>

        <div className="space-y-6">
          <Card className="p-6">
            <h3 className="font-semibold mb-4">Parameters</h3>
            <div className="space-y-6">
              {config.controls.map((control) => (
                <div key={control.name} className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <label className="text-sm font-medium">{control.label}</label>
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-6 w-6">
                            <HelpCircle className="h-3 w-3" />
                            <span className="sr-only">Parameter info</span>
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-80">
                          <p className="text-sm">{control.description}</p>
                        </PopoverContent>
                      </Popover>
                    </div>
                    <Badge variant="secondary" className="font-mono">
                      {params[control.name]?.toFixed(2) || control.defaultValue.toFixed(2)}
                    </Badge>
                  </div>
                  <Slider
                    value={[params[control.name] || control.defaultValue]}
                    onValueChange={(value) => handleParamChange(control.name, value)}
                    min={control.min}
                    max={control.max}
                    step={control.step}
                    className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
                  />
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="font-semibold mb-4">Metrics</h3>
            <div className="space-y-3">
              {config.metrics.map((metric) => (
                <div key={metric.name} className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">{metric.label}:</span>
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-6 w-6">
                          <HelpCircle className="h-3 w-3" />
                          <span className="sr-only">Metric info</span>
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-80">
                        <p className="text-sm">{metric.description}</p>
                      </PopoverContent>
                    </Popover>
                  </div>
                  <Badge variant="secondary" className={cn(
                    "font-mono",
                    (metric.name === 'accuracy' || metric.name === 'r2' || metric.name === 'precision' || metric.name === 'recall' || metric.name === 'f1' || metric.name === 'silhouette') && metrics[metric.name] > 0.8 ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300" : 
                    (metric.name === 'mse' || metric.name === 'rmse' || metric.name === 'mae') && metrics[metric.name] < 3 ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300" : ""
                  )}>
                    {metrics[metric.name]?.toFixed(2) || "0.00"}
                  </Badge>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <Card className="p-6">
        <Tabs defaultValue="explanation" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explanation">Explanation</TabsTrigger>
            <TabsTrigger value="code">Code</TabsTrigger>
            <TabsTrigger value="controls">Controls</TabsTrigger>
          </TabsList>
          <TabsContent value="explanation" className="mt-4 space-y-4">
            <p className="text-sm text-muted-foreground">
              {config.explanation}
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div className="border rounded-md p-4">
                <h4 className="text-sm font-medium mb-2">When to use {algorithm.title}</h4>
                <ul className="text-sm text-muted-foreground space-y-1 list-disc pl-4">
                  {algorithm.id === "regression" && (
                    <>
                      <li>Predicting continuous values (e.g., house prices, temperature)</li>
                      <li>Understanding relationships between variables</li>
                      <li>Forecasting trends based on historical data</li>
                    </>
                  )}
                  {algorithm.id === "classification" && (
                    <>
                      <li>Categorizing data into discrete classes</li>
                      <li>Email spam detection</li>
                      <li>Medical diagnosis</li>
                      <li>Customer churn prediction</li>
                    </>
                  )}
                  {algorithm.id === "clustering" && (
                    <>
                      <li>Customer segmentation</li>
                      <li>Anomaly detection</li>
                      <li>Document categorization</li>
                      <li>Image segmentation</li>
                    </>
                  )}
                  {algorithm.id === "eda" && (
                    <>
                      <li>Initial data understanding</li>
                      <li>Identifying patterns and relationships</li>
                      <li>Detecting outliers and anomalies</li>
                      <li>Checking assumptions before modeling</li>
                    </>
                  )}
                  {algorithm.id === "dimensionality" && (
                    <>
                      <li>Reducing computational complexity</li>
                      <li>Visualizing high-dimensional data</li>
                      <li>Removing noise and redundant features</li>
                      <li>Preventing overfitting</li>
                    </>
                  )}
                  {algorithm.id === "neural_networks" && (
                    <>
                      <li>Complex pattern recognition</li>
                      <li>Image and speech recognition</li>
                      <li>Natural language processing</li>
                      <li>When large amounts of data are available</li>
                    </>
                  )}
                  {algorithm.id === "nlp" && (
                    <>
                      <li>Text classification and sentiment analysis</li>
                      <li>Machine translation</li>
                      <li>Chatbots and virtual assistants</li>
                      <li>Information extraction from text</li>
                    </>
                  )}
                  {algorithm.id === "time_series" && (
                    <>
                      <li>Stock price prediction</li>
                      <li>Weather forecasting</li>
                      <li>Sales and demand forecasting</li>
                      <li>Analyzing seasonal patterns</li>
                    </>
                  )}
                  {algorithm.id === "reinforcement" && (
                    <>
                      <li>Game playing agents</li>
                      <li>Robotics and control systems</li>
                      <li>Resource management</li>
                      <li>Recommendation systems</li>
                    </>
                  )}
                  {algorithm.id === "computer_vision" && (
                    <>
                      <li>Image classification</li>
                      <li>Object detection and recognition</li>
                      <li>Facial recognition</li>
                      <li>Medical image analysis</li>
                    </>
                  )}
                  {algorithm.id === "ensemble" && (
                    <>
                      <li>When high prediction accuracy is crucial</li>
                      <li>Handling complex datasets with noise</li>
                      <li>Competitions and real-world applications</li>
                      <li>When individual models have complementary strengths</li>
                    </>
                  )}
                </ul>
              </div>
              <div className="border rounded-md p-4">
                <h4 className="text-sm font-medium mb-2">Key Concepts</h4>
                <ul className="text-sm text-muted-foreground space-y-1 list-disc pl-4">
                  {algorithm.id === "regression" && (
                    <>
                      <li><span className="font-medium">Coefficients:</span> Weights assigned to features</li>
                      <li><span className="font-medium">Intercept:</span> Base value when all features are zero</li>
                      <li><span className="font-medium">R²:</span> Proportion of variance explained by the model</li>
                      <li><span className="font-medium">Residuals:</span> Differences between predicted and actual values</li>
                    </>
                  )}
                  {algorithm.id === "classification" && (
                    <>
                      <li><span className="font-medium">Decision Boundary:</span> Surface that separates different classes</li>
                      <li><span className="font-medium">Probability:</span> Likelihood of belonging to a class</li>
                      <li><span className="font-medium">Confusion Matrix:</span> Table showing correct and incorrect predictions</li>
                      <li><span className="font-medium">ROC Curve:</span> Plot of true positive rate vs. false positive rate</li>
                    </>
                  )}
                  {algorithm.id === "clustering" && (
                    <>
                      <li><span className="font-medium">Centroids:</span> Center points of clusters</li>
                      <li><span className="font-medium">Inertia:</span> Sum of distances to nearest centroid</li>
                      <li><span className="font-medium">Silhouette Score:</span> Measure of cluster separation</li>
                      <li><span className="font-medium">Hierarchical Clustering:</span> Nested clusters approach</li>
                    </>
                  )}
                  {algorithm.id === "eda" && (
                    <>
                      <li><span className="font-medium">Descriptive Statistics:</span> Mean, median, standard deviation</li>
                      <li><span className="font-medium">Data Visualization:</span> Histograms, scatter plots, box plots</li>
                      <li><span className="font-medium">Correlation:</span> Relationship strength between variables</li>
                      <li><span className="font-medium">Outliers:</span> Data points that differ significantly from others</li>
                    </>
                  )}
                  {algorithm.id === "dimensionality" && (
                    <>
                      <li><span className="font-medium">Principal Components:</span> New uncorrelated variables</li>
                      <li><span className="font-medium">Explained Variance:</span> Information retained after reduction</li>
                      <li><span className="font-medium">Manifold Learning:</span> Finding lower-dimensional structure</li>
                      <li><span className="font-medium">Feature Selection:</span> Choosing most important variables</li>
                    </>
                  )}
                  {algorithm.id === "neural_networks" && (
                    <>
                      <li><span className="font-medium">Neurons:</span> Basic computational units</li>
                      <li><span className="font-medium">Activation Functions:</span> Introduce non-linearity</li>
                      <li><span className="font-medium">Backpropagation:</span> Algorithm for updating weights</li>
                      <li><span className="font-medium">Overfitting:</span> Model learns noise instead of patterns</li>
                    </>
                  )}
                  {algorithm.id === "nlp" && (
                    <>
                      <li><span className="font-medium">Tokenization:</span> Breaking text into words or subwords</li>
                      <li><span className="font-medium">Embeddings:</span> Vector representations of words</li>
                      <li><span className="font-medium">Sequence Models:</span> RNNs, LSTMs, Transformers</li>
                      <li><span className="font-medium">Transfer Learning:</span> Using pre-trained language models</li>
                    </>
                  )}
                  {algorithm.id === "time_series" && (
                    <>
                      <li><span className="font-medium">Trend:</span> Long-term direction</li>
                      <li><span className="font-medium">Seasonality:</span> Regular patterns over fixed periods</li>
                      <li><span className="font-medium">Stationarity:</span> Statistical properties don't change over time</li>
                      <li><span className="font-medium">Autocorrelation:</span> Correlation of a signal with itself</li>
                    </>
                  )}
                  {algorithm.id === "reinforcement" && (
                    <>
                      <li><span className="font-medium">Agent:</span> Entity that takes actions</li>
                      <li><span className="font-medium">Environment:</span> World the agent interacts with</li>
                      <li><span className="font-medium">Reward:</span> Feedback signal for actions</li>
                      <li><span className="font-medium">Policy:</span> Strategy for selecting actions</li>
                    </>
                  )}
                  {algorithm.id === "computer_vision" && (
                    <>
                      <li><span className="font-medium">Convolutional Layers:</span> Extract features from images</li>
                      <li><span className="font-medium">Pooling:</span> Reduce spatial dimensions</li>
                      <li><span className="font-medium">Feature Maps:</span> Activations of convolutional filters</li>
                      <li><span className="font-medium">Transfer Learning:</span> Using pre-trained models</li>
                    </>
                  )}
                  {algorithm.id === "ensemble" && (
                    <>
                      <li><span className="font-medium">Bagging:</span> Training models on random subsets (e.g., Random Forest)</li>
                      <li><span className="font-medium">Boosting:</span> Sequential training focusing on errors</li>
                      <li><span className="font-medium">Stacking:</span> Using predictions as features for meta-learner</li>
                      <li><span className="font-medium">Voting:</span> Combining predictions through voting</li>
                    </>
                  )}
                </ul>
              </div>
            </div>
          </TabsContent>
          <TabsContent value="code" className="mt-4">
            <pre className="text-sm bg-muted p-4 rounded-lg overflow-x-auto">
              <code>{config.code}</code>
            </pre>
          </TabsContent>
          <TabsContent value="controls" className="mt-4">
            <div className="grid gap-4 md:grid-cols-3">
              <Button variant="outline" onClick={() => {
                const initialParams: Record<string, number> = {};
                config.controls.forEach(control => {
                  initialParams[control.name] = control.defaultValue;
                });
                setParams(initialParams);
              }}>
                Reset Parameters
              </Button>
              <Button variant="outline" onClick={() => {
                const newParams = { ...params };
                config.controls.forEach(control => {
                  newParams[control.name] = params[control.name] + (Math.random() * 0.2 - 0.1) * (control.max - control.min);
                  // Keep within bounds
                  newParams[control.name] = Math.min(Math.max(newParams[control.name], control.min), control.max);
                });
                setParams(newParams);
              }}>
                Randomize Slightly
              </Button>
              <Button variant="outline" onClick={() => {
                const newParams = { ...params };
                config.controls.forEach(control => {
                  newParams[control.name] = control.min + Math.random() * (control.max - control.min);
                });
                setParams(newParams);
              }}>
                Randomize Completely
              </Button>
            </div>
            
            <Separator className="my-4" />
            
            <div className="space-y-4">
              <h4 className="text-sm font-medium">Parameter Effects</h4>
              <div className="text-sm text-muted-foreground space-y-2">
                {algorithm.id === "regression" && (
                  <>
                    <p><span className="font-medium">Weight:</span> Controls the slope of the line. Higher values make the line steeper, while negative values reverse the direction.</p>
                    <p><span className="font-medium">Bias:</span> Shifts the line up or down without changing its slope.</p>
                    <p><span className="font-medium">Noise Level:</span> Adds random variation to the data points, simulating real-world measurement error.</p>
                  </>
                )}
                {algorithm.id === "classification" && (
                  <>
                    <p><span className="font-medium">Decision Boundary:</span> Threshold that determines class separation. Adjusting this shifts the boundary between classes.</p>
                    <p><span className="font-medium">Feature Weights:</span> Control the importance of each feature in making classification decisions.</p>
                  </>
                )}
                {algorithm.id === "clustering" && (
                  <>
                    <p><span className="font-medium">Number of Clusters:</span> Determines how many groups the algorithm will identify. Too few may combine distinct groups, while too many may split natural clusters.</p>
                    <p><span className="font-medium">Max Iterations:</span> Limits how long the algorithm runs. Higher values allow more time to find optimal clusters but may take longer.</p>
                    <p><span className="font-medium">Random Seed:</span> Changes the initial cluster centers, which can lead to different final clusters.</p>
                  </>
                )}
                {algorithm.id === "neural_networks" && (
                  <>
                    <p><span className="font-medium">Learning Rate:</span> Controls how quickly the model adapts to new information. Higher values may converge faster but risk overshooting the optimal solution.</p>
                    <p><span className="font-medium">Hidden Layers:</span> More layers allow the network to learn more complex patterns but increase training time and risk of overfitting.</p>
                    <p><span className="font-medium">Neurons Per Layer:</span> More neurons increase model capacity but may lead to overfitting on small datasets.</p>
                  </>
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  )
}
