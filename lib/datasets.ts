export interface Dataset {
    id: string
    name: string
    description: string
    previewCode: string
    algorithm: string
    url?: string
  }
  
  const datasets: Dataset[] = [
    // EDA Datasets
    {
      id: "tips",
      name: "Restaurant Tips",
      description: "Dataset containing restaurant tips and related information",
      algorithm: "eda",
      url: "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
      previewCode: `# Import necessary libraries
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # Load the restaurant tips dataset
  url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
  df = pd.read_csv(url)
  
  # Display the first 5 rows
  print("First 5 rows of the tips dataset:")
  print(df.head())
  
  # Display basic information
  print("\nDataset information:")
  print(df.info())
  
  # Display summary statistics
  print("\nSummary statistics:")
  print(df.describe())`,
    },
    {
      id: "iris",
      name: "Iris Flower Dataset",
      description: "Classic dataset containing measurements of iris flowers",
      algorithm: "eda",
      url: "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
      previewCode: `# Import necessary libraries
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # Load the Iris dataset
  url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
  df = pd.read_csv(url)
  
  # Display the first 5 rows
  print("First 5 rows of the Iris dataset:")
  print(df.head())
  
  # Display basic information
  print("\nDataset information:")
  print(df.info())
  
  # Display summary statistics
  print("\nSummary statistics:")
  print(df.describe())`,
    },
    {
      id: "diamonds",
      name: "Diamonds Dataset",
      description: "Dataset containing prices and attributes of diamonds",
      algorithm: "eda",
      url: "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
      previewCode: `# Import necessary libraries
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # Load the Diamonds dataset
  url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
  df = pd.read_csv(url)
  
  # Display the first 5 rows
  print("First 5 rows of the Diamonds dataset:")
  print(df.head())
  
  # Display basic information
  print("\nDataset information:")
  print(df.info())
  
  # Display summary statistics
  print("\nSummary statistics:")
  print(df.describe())`,
    },
  
    // Linear Regression Datasets
    {
      id: "housing",
      name: "Housing Dataset",
      description: "Dataset containing house prices and features",
      algorithm: "linear-regression",
      url: "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv",
      previewCode: `# Import necessary libraries
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # Load the Housing dataset
  url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
  df = pd.read_csv(url)
  
  # Display the first 5 rows
  print("First 5 rows of the Housing dataset:")
  print(df.head())
  
  # Display basic information
  print("\nDataset information:")
  print(df.info())
  
  # Display summary statistics
  print("\nSummary statistics:")
  print(df.describe())
  
  # Visualize the distribution of prices
  plt.figure(figsize=(10, 6))
  plt.hist(df['price'], bins=30, color='skyblue', edgecolor='black')
  plt.title('Distribution of House Prices')
  plt.xlabel('Price')
  plt.ylabel('Frequency')
  plt.grid(True, alpha=0.3)`,
    },
    // SVM Datasets
    {
      id: "iris_svm",
      name: "Iris Dataset for SVM",
      description: "Iris flower dataset for classification",
      algorithm: "svm",
      previewCode: `# Import necessary libraries
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.datasets import load_iris
  
  # Load the Iris dataset
  iris = load_iris()
  X = iris.data
  y = iris.target
  feature_names = iris.feature_names
  target_names = iris.target_names
  
  # Create a DataFrame for easier exploration
  df = pd.DataFrame(X, columns=feature_names)
  df['target'] = y
  df['species'] = df['target'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})
  
  # Display the first 5 rows
  print("First 5 rows of the Iris dataset:")
  print(df.head())
  
  # Display basic information
  print("\nDataset information:")
  print(df.info())
  
  # Display summary statistics
  print("\nSummary statistics:")
  print(df.describe())
  
  # Visualize the data
  plt.figure(figsize=(10, 6))
  sns.scatterplot(x=df[feature_names[0]], y=df[feature_names[1]], hue=df['species'], palette='viridis')
  plt.title('Iris Dataset: Sepal Length vs Sepal Width')
  plt.xlabel(feature_names[0])
  plt.ylabel(feature_names[1])`,
    },
    {
      id: "breast_cancer",
      name: "Breast Cancer Dataset",
      description: "Dataset for breast cancer classification",
      algorithm: "svm",
      previewCode: `# Import necessary libraries
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.datasets import load_breast_cancer
  
  # Load the Breast Cancer dataset
  cancer = load_breast_cancer()
  X = cancer.data
  y = cancer.target
  feature_names = cancer.feature_names
  target_names = cancer.target_names
  
  # Create a DataFrame for easier exploration
  df = pd.DataFrame(X, columns=feature_names)
  df['target'] = y
  df['diagnosis'] = df['target'].map({0: target_names[0], 1: target_names[1]})
  
  # Display the first 5 rows
  print("First 5 rows of the Breast Cancer dataset:")
  print(df.head())
  
  # Display basic information
  print("\nDataset information:")
  print(df.info())
  
  # Display summary statistics
  print("\nSummary statistics:")
  print(df.describe())
  
  # Visualize the data
  plt.figure(figsize=(10, 6))
  sns.scatterplot(x=df[feature_names[0]], y=df[feature_names[1]], hue=df['diagnosis'], palette='viridis')
  plt.title('Breast Cancer Dataset: Feature Visualization')
  plt.xlabel(feature_names[0])
  plt.ylabel(feature_names[1])`,
    },
  
    // K-means Datasets
    {
      id: "mall_customers",
      name: "Mall Customers Dataset",
      description: "Dataset for customer segmentation",
      algorithm: "kmeans",
      url: "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv",
      previewCode: `# Import necessary libraries
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # Load the Mall Customers dataset
  url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"
  df = pd.read_csv(url)
  
  # Display the first 5 rows
  print("First 5 rows of the Mall Customers dataset:")
  print(df.head())
  
  # Display basic information
  print("\nDataset information:")
  print(df.info())
  
  # Display summary statistics
  print("\nSummary statistics:")
  print(df.describe())
  
  # Visualize the data
  plt.figure(figsize=(10, 6))
  sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Gender'])
  plt.title('Mall Customers: Income vs Spending Score')
  plt.xlabel('Annual Income (k$)')
  plt.ylabel('Spending Score (1-100)')`,
    },
    {
      id: "synthetic_clusters",
      name: "Synthetic Clusters Dataset",
      description: "Synthetic dataset with well-defined clusters",
      algorithm: "kmeans",
      previewCode: `# Import necessary libraries
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.datasets import make_blobs
  
  # Generate synthetic data with well-defined clusters
  X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
  
  # Create a DataFrame
  df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
  df['Cluster'] = y
  
  # Display the first 5 rows
  print("First 5 rows of the Synthetic Clusters dataset:")
  print(df.head())
  
  # Display basic information
  print("\nDataset information:")
  print(df.info())
  
  # Display summary statistics
  print("\nSummary statistics:")
  print(df.describe())
  
  # Visualize the data
  plt.figure(figsize=(10, 6))
  sns.scatterplot(x=df['Feature1'], y=df['Feature2'], hue=df['Cluster'], palette='viridis')
  plt.title('Synthetic Clusters Dataset')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')`,
    },
  ]
  
  export function getDatasetsByAlgorithm(algorithm: string): Dataset[] {
    return datasets.filter((dataset) => dataset.algorithm === algorithm)
  }
  
  export function getDatasetById(id: string): Dataset | undefined {
    return datasets.find((dataset) => dataset.id === id)
  }
  
  