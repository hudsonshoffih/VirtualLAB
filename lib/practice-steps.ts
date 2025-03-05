import type { PracticeStep } from "./types"

// EDA Practice Steps
const edaPracticeSteps: PracticeStep[] = [
  {
    title: "Loading and Exploring Data",
    instruction: `
      <p>In this step, you'll learn how to load a dataset and perform basic exploration.</p>
      <p>Use pandas to load the dataset and display its first few rows.</p>
      <ol>
        <li>Import pandas as pd</li>
        <li>Load the dataset using pd.read_csv</li>
        <li>Display the first 5 rows using the head() method</li>
      </ol>
    `,
    starterCode: `# Import pandas
import pandas as pd

# Load the dataset (we'll use a sample dataset about housing prices)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"

# Your code here to load the dataset and display the first 5 rows
`,
    hints: [
      "Use pd.read_csv(url) to load the dataset",
      "Use df.head() to display the first 5 rows",
      "You can use print() to display text in the output",
    ],
    resources: [
      { title: "Pandas Documentation", url: "https://pandas.pydata.org/docs/" },
      {
        title: "DataFrame.head() Method",
        url: "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html",
      },
    ],
  },
  {
    title: "Data Summary Statistics",
    instruction: `
      <p>Now that you've loaded the data, let's examine its statistical properties.</p>
      <p>Use pandas methods to get a summary of the dataset.</p>
      <ol>
        <li>Check the shape of the dataset (rows and columns)</li>
        <li>Get basic information about the dataset using info()</li>
        <li>Calculate summary statistics using describe()</li>
      </ol>
    `,
    starterCode: `# Import pandas
import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
df = pd.read_csv(url)

# Your code here to examine the dataset
# 1. Check the shape of the dataset
# 2. Get basic information about the dataset
# 3. Calculate summary statistics
`,
    hints: [
      "Use df.shape to get the number of rows and columns",
      "Use df.info() to get information about the data types and missing values",
      "Use df.describe() to get summary statistics for numerical columns",
    ],
    resources: [
      {
        title: "DataFrame.info() Method",
        url: "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html",
      },
      {
        title: "DataFrame.describe() Method",
        url: "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html",
      },
    ],
  },
  {
    title: "Handling Missing Values",
    instruction: `
      <p>Missing values can affect your analysis. Let's check for and handle missing values in the dataset.</p>
      <ol>
        <li>Check for missing values in each column</li>
        <li>Calculate the percentage of missing values</li>
        <li>Fill missing values with appropriate methods</li>
      </ol>
    `,
    starterCode: `# Import pandas
import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
df = pd.read_csv(url)

# Your code here to check and handle missing values
# 1. Check for missing values in each column
# 2. Calculate the percentage of missing values
# 3. Fill missing values with appropriate methods
`,
    hints: [
      "Use df.isnull().sum() to count missing values in each column",
      "Calculate percentage as (missing_count / total_rows) * 100",
      "Use df.fillna() to fill missing values",
    ],
    resources: [
      { title: "Working with Missing Data", url: "https://pandas.pydata.org/docs/user_guide/missing_data.html" },
      {
        title: "DataFrame.fillna() Method",
        url: "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html",
      },
    ],
  },
  {
    title: "Data Visualization",
    instruction: `
      <p>Visualizing data helps in understanding patterns and relationships. Let's create some basic visualizations.</p>
      <ol>
        <li>Import matplotlib and seaborn</li>
        <li>Create a histogram of a numerical column</li>
        <li>Create a scatter plot to explore relationships between variables</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for better visualizations
sns.set(style="whitegrid")

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
df = pd.read_csv(url)

# Your code here to create visualizations
# 1. Create a histogram of a numerical column
# 2. Create a scatter plot to explore relationships between variables
`,
    hints: [
      "Use plt.figure(figsize=(10, 6)) to set the figure size",
      "Use sns.histplot(df['column_name']) to create a histogram",
      "Use sns.scatterplot(x='column1', y='column2', data=df) for scatter plots",
    ],
    resources: [
      { title: "Seaborn Documentation", url: "https://seaborn.pydata.org/" },
      { title: "Matplotlib Documentation", url: "https://matplotlib.org/stable/contents.html" },
    ],
  },
  {
    title: "Correlation Analysis",
    instruction: `
      <p>Correlation analysis helps identify relationships between variables. Let's calculate and visualize correlations.</p>
      <ol>
        <li>Calculate the correlation matrix for numerical columns</li>
        <li>Visualize the correlation matrix using a heatmap</li>
        <li>Interpret the results</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for better visualizations
sns.set(style="whitegrid")

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
df = pd.read_csv(url)

# Your code here to perform correlation analysis
# 1. Calculate the correlation matrix
# 2. Visualize the correlation matrix using a heatmap
# 3. Add your interpretation as comments
`,
    hints: [
      "Use df.corr() to calculate the correlation matrix",
      "Use sns.heatmap(df.corr(), annot=True) to create a heatmap",
      "Look for strong positive (close to 1) or negative (close to -1) correlations",
    ],
    resources: [
      {
        title: "DataFrame.corr() Method",
        url: "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html",
      },
      { title: "Seaborn Heatmap", url: "https://seaborn.pydata.org/generated/seaborn.heatmap.html" },
    ],
  },
]

// Linear Regression Practice Steps
const linearRegressionPracticeSteps: PracticeStep[] = [
  {
    title: "Data Preparation",
    instruction: `
      <p>In this step, you'll prepare data for a linear regression model.</p>
      <p>We'll use a housing dataset to predict house prices based on features like area, bedrooms, etc.</p>
      <ol>
        <li>Import necessary libraries</li>
        <li>Load the dataset</li>
        <li>Select features (X) and target variable (y)</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
df = pd.read_csv(url)

# Your code here to prepare the data
# 1. Display the first few rows to understand the data
# 2. Select features (X) and target variable (y)
`,
    hints: [
      "Use df.head() to view the first few rows",
      "For features (X), select columns that might influence the price",
      "For target (y), select the 'price' column",
    ],
    resources: [
      { title: "Scikit-learn Documentation", url: "https://scikit-learn.org/stable/" },
      {
        title: "Linear Regression Guide",
        url: "https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares",
      },
    ],
  },
  {
    title: "Train-Test Split",
    instruction: `
      <p>Before building a model, it's important to split the data into training and testing sets.</p>
      <p>This allows us to evaluate how well our model generalizes to new data.</p>
      <ol>
        <li>Split the data into training and testing sets</li>
        <li>Print the shapes of the resulting sets</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
df = pd.read_csv(url)

# Select features (X) and target variable (y)
# Let's use 'area', 'bedrooms', 'bathrooms', and 'stories' to predict 'price'
X = df[['area', 'bedrooms', 'bathrooms', 'stories']]
y = df['price']

# Your code here to split the data into training and testing sets
# Use train_test_split with test_size=0.2 and random_state=42
`,
    hints: [
      "Use train_test_split(X, y, test_size=0.2, random_state=42)",
      "This will create X_train, X_test, y_train, y_test",
      "Print the shape of each to verify the split",
    ],
    resources: [
      {
        title: "Train Test Split",
        url: "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
      },
    ],
  },
  {
    title: "Building the Linear Regression Model",
    instruction: `
      <p>Now let's build and train a linear regression model using scikit-learn.</p>
      <ol>
        <li>Import the LinearRegression class</li>
        <li>Create a model instance</li>
        <li>Fit the model to the training data</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
df = pd.read_csv(url)

# Select features (X) and target variable (y)
X = df[['area', 'bedrooms', 'bathrooms', 'stories']]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Your code here to build and train the linear regression model
`,
    hints: [
      "Create a model with model = LinearRegression()",
      "Fit the model with model.fit(X_train, y_train)",
      "This trains the model on your training data",
    ],
    resources: [
      {
        title: "Linear Regression",
        url: "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
      },
    ],
  },
  {
    title: "Making Predictions and Evaluating the Model",
    instruction: `
      <p>After training the model, let's use it to make predictions and evaluate its performance.</p>
      <ol>
        <li>Make predictions on the test set</li>
        <li>Calculate the model's performance metrics</li>
        <li>Visualize the actual vs. predicted values</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
df = pd.read_csv(url)

# Select features (X) and target variable (y)
X = df[['area', 'bedrooms', 'bathrooms', 'stories']]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Your code here to make predictions and evaluate the model
# 1. Make predictions on the test set
# 2. Calculate MSE and R² score
# 3. Create a scatter plot of actual vs. predicted values
`,
    hints: [
      "Use model.predict(X_test) to make predictions",
      "Calculate MSE with mean_squared_error(y_test, y_pred)",
      "Calculate R² with r2_score(y_test, y_pred)",
      "Use plt.scatter() to plot actual vs. predicted values",
    ],
    resources: [
      { title: "Model Evaluation", url: "https://scikit-learn.org/stable/modules/model_evaluation.html" },
      {
        title: "Regression Metrics",
        url: "https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics",
      },
    ],
  },
  {
    title: "Interpreting the Model Coefficients",
    instruction: `
      <p>Linear regression models are interpretable. Let's examine the coefficients to understand the relationship between features and the target.</p>
      <ol>
        <li>Extract and print the model coefficients</li>
        <li>Create a DataFrame to display features and their coefficients</li>
        <li>Visualize the coefficients using a bar plot</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
df = pd.read_csv(url)

# Select features (X) and target variable (y)
X = df[['area', 'bedrooms', 'bathrooms', 'stories']]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Your code here to interpret the model coefficients
# 1. Extract and print the model coefficients
# 2. Create a DataFrame to display features and their coefficients
# 3. Visualize the coefficients using a bar plot
`,
    hints: [
      "Access coefficients with model.coef_",
      "Access intercept with model.intercept_",
      "Create a DataFrame with pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})",
      "Use plt.barh() to create a horizontal bar plot of coefficients",
    ],
    resources: [
      {
        title: "Linear Models Coefficients",
        url: "https://scikit-learn.org/stable/modules/linear_model.html#linear-model",
      },
      {
        title: "Feature Importance",
        url: "https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html",
      },
    ],
  },
]

// SVM Practice Steps
const svmPracticeSteps: PracticeStep[] = [
  {
    title: "Data Preparation for SVM",
    instruction: `
      <p>In this step, you'll prepare data for a Support Vector Machine (SVM) classifier.</p>
      <p>We'll use the Iris dataset, a classic dataset for classification problems.</p>
      <ol>
        <li>Import necessary libraries</li>
        <li>Load the Iris dataset</li>
        <li>Explore the dataset structure</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Your code here to load and explore the Iris dataset
# 1. Load the dataset
# 2. Convert to a pandas DataFrame
# 3. Explore the dataset structure
`,
    hints: [
      "Use load_iris() to get the dataset",
      "Create a DataFrame with pd.DataFrame(data=iris.data, columns=iris.feature_names)",
      "Add the target column with df['target'] = iris.target",
    ],
    resources: [
      { title: "Scikit-learn Datasets", url: "https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset" },
      { title: "SVM Documentation", url: "https://scikit-learn.org/stable/modules/svm.html" },
    ],
  },
  {
    title: "Visualizing the Data",
    instruction: `
      <p>Before building an SVM model, it's helpful to visualize the data to understand the classification problem.</p>
      <ol>
        <li>Create a scatter plot of two features</li>
        <li>Color the points by their class</li>
        <li>Add a legend and labels</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Your code here to create a scatter plot
# 1. Create a scatter plot of sepal length vs. sepal width
# 2. Color the points by their class
# 3. Add a legend and labels
`,
    hints: [
      "Use plt.figure(figsize=(10, 6)) to set the figure size",
      "Use plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'])",
      "Add labels with plt.xlabel() and plt.ylabel()",
      "Add a legend with plt.legend()",
    ],
    resources: [
      { title: "Matplotlib Scatter", url: "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html" },
      { title: "Data Visualization", url: "https://scikit-learn.org/stable/visualizations.html" },
    ],
  },
  {
    title: "Building an SVM Classifier",
    instruction: `
      <p>Now let's build and train an SVM classifier using scikit-learn.</p>
      <ol>
        <li>Split the data into training and testing sets</li>
        <li>Create an SVM classifier with a linear kernel</li>
        <li>Train the classifier on the training data</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Your code here to build and train an SVM classifier
# 1. Split the data into training and testing sets
# 2. Scale the features (important for SVM)
# 3. Create an SVM classifier with a linear kernel
# 4. Train the classifier
`,
    hints: [
      "Use train_test_split(X, y, test_size=0.3, random_state=42)",
      "Scale features with scaler = StandardScaler() and X_train_scaled = scaler.fit_transform(X_train)",
      "Create an SVM with svm = SVC(kernel='linear')",
      "Train with svm.fit(X_train_scaled, y_train)",
    ],
    resources: [
      { title: "SVC Documentation", url: "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html" },
      {
        title: "Feature Scaling",
        url: "https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling",
      },
    ],
  },
  {
    title: "Evaluating the SVM Model",
    instruction: `
      <p>After training the SVM model, let's evaluate its performance on the test set.</p>
      <ol>
        <li>Make predictions on the test set</li>
        <li>Calculate accuracy, precision, recall, and F1-score</li>
        <li>Create a confusion matrix</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)

# Your code here to evaluate the model
# 1. Make predictions on the test set
# 2. Calculate accuracy, precision, recall, and F1-score
# 3. Create and visualize a confusion matrix
`,
    hints: [
      "Use y_pred = svm.predict(X_test_scaled) to make predictions",
      "Calculate accuracy with accuracy_score(y_test, y_pred)",
      "Get detailed metrics with print(classification_report(y_test, y_pred))",
      "Create a confusion matrix with confusion_matrix(y_test, y_pred)",
    ],
    resources: [
      { title: "Model Evaluation", url: "https://scikit-learn.org/stable/modules/model_evaluation.html" },
      {
        title: "Classification Metrics",
        url: "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics",
      },
    ],
  },
  {
    title: "Visualizing Decision Boundaries",
    instruction: `
      <p>One of the advantages of SVM is that we can visualize the decision boundaries it creates.</p>
      <p>Let's create a visualization of the SVM decision boundaries for two features.</p>
      <ol>
        <li>Select two features for visualization</li>
        <li>Train an SVM on these two features</li>
        <li>Create a mesh grid and visualize the decision boundaries</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # We'll use only the first two features for visualization
y = iris.target

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the SVM
svm = SVC(kernel='linear')
svm.fit(X_scaled, y)

# Your code here to visualize the decision boundaries
# 1. Create a mesh grid
# 2. Predict the class for each point in the mesh
# 3. Plot the decision boundaries and the data points
`,
    hints: [
      "Create a mesh with np.meshgrid(np.linspace(X_scaled[:, 0].min()-1, X_scaled[:, 0].max()+1, 100), np.linspace(X_scaled[:, 1].min()-1, X_scaled[:, 1].max()+1, 100))",
      "Reshape and predict with Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)",
      "Plot with plt.contourf(xx, yy, Z, alpha=0.3) and plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)",
    ],
    resources: [
      { title: "SVM Visualization", url: "https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html" },
      {
        title: "Decision Boundaries",
        url: "https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html",
      },
    ],
  },
]

// K-Means Clustering Practice Steps
const kmeansPracticeSteps: PracticeStep[] = [
  {
    title: "Data Preparation for Clustering",
    instruction: `
      <p>In this step, you'll prepare data for K-means clustering.</p>
      <p>We'll use a simple dataset to demonstrate the clustering process.</p>
      <ol>
        <li>Import necessary libraries</li>
        <li>Create or load a dataset suitable for clustering</li>
        <li>Visualize the data to understand its structure</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Your code here to create and visualize a dataset
# 1. Create a synthetic dataset with make_blobs
# 2. Convert to a pandas DataFrame
# 3. Visualize the data points
`,
    hints: [
      "Use make_blobs(n_samples=300, centers=4, random_state=42) to create a dataset",
      "Create a DataFrame with pd.DataFrame(data=X, columns=['Feature1', 'Feature2'])",
      "Visualize with plt.scatter(X[:, 0], X[:, 1])",
    ],
    resources: [
      {
        title: "make_blobs Documentation",
        url: "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html",
      },
      { title: "K-means Documentation", url: "https://scikit-learn.org/stable/modules/clustering.html#k-means" },
    ],
  },
  {
    title: "Implementing K-means Clustering",
    instruction: `
      <p>Now let's implement K-means clustering using scikit-learn.</p>
      <ol>
        <li>Import the KMeans class</li>
        <li>Create a K-means model with an appropriate number of clusters</li>
        <li>Fit the model to the data</li>
        <li>Get the cluster assignments and centroids</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Your code here to implement K-means clustering
# 1. Create a K-means model with 4 clusters
# 2. Fit the model to the data
# 3. Get the cluster assignments and centroids
`,
    hints: [
      "Create a model with kmeans = KMeans(n_clusters=4, random_state=42)",
      "Fit the model with kmeans.fit(X)",
      "Get cluster assignments with kmeans.labels_",
      "Get centroids with kmeans.cluster_centers_",
    ],
    resources: [
      {
        title: "KMeans Documentation",
        url: "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
      },
    ],
  },
  {
    title: "Visualizing Clusters",
    instruction: `
      <p>After applying K-means, let's visualize the clusters and centroids.</p>
      <ol>
        <li>Create a scatter plot of the data points</li>
        <li>Color the points according to their cluster assignments</li>
        <li>Mark the cluster centroids</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Your code here to visualize the clusters and centroids
# 1. Create a scatter plot of the data points, colored by cluster
# 2. Mark the cluster centroids
# 3. Add a legend and labels
`,
    hints: [
      "Use plt.figure(figsize=(10, 6)) to set the figure size",
      "Use plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis') to plot colored points",
      "Use plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red') to mark centroids",
    ],
    resources: [
      { title: "Matplotlib Scatter", url: "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html" },
      {
        title: "K-means Visualization",
        url: "https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html",
      },
    ],
  },
  {
    title: "Finding the Optimal Number of Clusters",
    instruction: `
      <p>One challenge in K-means is determining the optimal number of clusters. Let's use the Elbow Method to find it.</p>
      <ol>
        <li>Run K-means with different numbers of clusters</li>
        <li>Calculate the inertia (sum of squared distances) for each</li>
        <li>Plot the inertia vs. number of clusters</li>
        <li>Identify the "elbow" point</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Your code here to find the optimal number of clusters
# 1. Create a range of cluster numbers to try
# 2. Calculate inertia for each number of clusters
# 3. Plot the inertia vs. number of clusters
# 4. Identify the "elbow" point
`,
    hints: [
      "Create a range with range(1, 11)",
      "Calculate inertia with KMeans(n_clusters=k).fit(X).inertia_",
      "Plot with plt.plot(range(1, 11), inertias, marker='o')",
      "Look for the point where the inertia starts decreasing more slowly",
    ],
    resources: [
      {
        title: "Elbow Method",
        url: "https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_stability_low_dim_dense.html",
      },
      { title: "Selecting Number of Clusters", url: "https://scikit-learn.org/stable/modules/clustering.html#k-means" },
    ],
  },
  {
    title: "Evaluating Clustering Quality",
    instruction: `
      <p>Let's evaluate the quality of our clustering using metrics like silhouette score.</p>
      <ol>
        <li>Import necessary evaluation metrics</li>
        <li>Calculate the silhouette score for different numbers of clusters</li>
        <li>Plot the silhouette scores</li>
        <li>Interpret the results</li>
      </ol>
    `,
    starterCode: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Your code here to evaluate clustering quality
# 1. Calculate silhouette scores for different numbers of clusters
# 2. Plot the silhouette scores
# 3. Interpret the results (add comments)
`,
    hints: [
      "Calculate silhouette score with silhouette_score(X, kmeans.labels_)",
      "Higher silhouette scores indicate better-defined clusters",
      "The optimal number of clusters often has the highest silhouette score",
    ],
    resources: [
      {
        title: "Silhouette Score",
        url: "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html",
      },
      {
        title: "Clustering Metrics",
        url: "https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation",
      },
    ],
  },
]

// Map algorithm slugs to their practice steps
const practiceStepsMap: Record<string, PracticeStep[]> = {
  eda: edaPracticeSteps,
  "linear-regression": linearRegressionPracticeSteps,
  svm: svmPracticeSteps,
  kmeans: kmeansPracticeSteps,
}

export function getPracticeSteps(algorithmSlug: string): PracticeStep[] {
  return practiceStepsMap[algorithmSlug] || []
}

