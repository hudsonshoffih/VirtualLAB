export interface Dataset {
    id: string
    name: string
    description: string
    algorithm: string
    previewCode: string
  }
  
  export const datasets: Dataset[] = [
    // Linear Regression Dataset
    {
      id: "housing",
      name: "Housing Prices",
      description: "Boston housing prices dataset with features like crime rate, room numbers, etc.",
      algorithm: "linear-regression",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load the dataset\ndf = pd.read_csv('housing.csv')\n\n# Display the first 5 rows\nprint(df.head())\n\n# Basic statistics\nprint(df.describe())\n\n# Correlation heatmap\nplt.figure(figsize=(10, 8))\nsns.heatmap(df.corr(), annot=True, cmap='coolwarm')\nplt.title('Feature Correlation')\nplt.show()",
    },
    {
      id: "salary",
      name: "Salary vs Experience",
      description: "Dataset showing relationship between years of experience and salary.",
      algorithm: "linear-regression",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\n\n# Load the dataset\ndf = pd.read_csv('salary.csv')\n\n# Display the first 5 rows\nprint(df.head())\n\n# Visualize the relationship\nplt.scatter(df['YearsExperience'], df['Salary'])\nplt.xlabel('Years of Experience')\nplt.ylabel('Salary')\nplt.title('Salary vs Experience')\nplt.show()",
    },
    {
      id: "advertising",
      name: "Advertising Impact",
      description: "Dataset showing impact of TV, radio, and newspaper advertising on sales.",
      algorithm: "linear-regression",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load the dataset\ndf = pd.read_csv('advertising.csv')\n\n# Display the first 5 rows\nprint(df.head())\n\n# Create subplots for each feature\nfig, axes = plt.subplots(1, 3, figsize=(15, 5))\n\nsns.regplot(x='TV', y='Sales', data=df, ax=axes[0])\nsns.regplot(x='Radio', y='Sales', data=df, ax=axes[1])\nsns.regplot(x='Newspaper', y='Sales', data=df, ax=axes[2])\n\nplt.tight_layout()\nplt.show()",
    },
    {
      id: "student-performance",
      name: "Student Performance",
      description: "Dataset showing relationship between study hours and exam scores.",
      algorithm: "linear-regression",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# Load the dataset\ndf = pd.read_csv('student_performance.csv')\n\n# Display the first 5 rows\nprint(df.head())\n\n# Visualize the relationship\nplt.scatter(df['Hours'], df['Scores'])\nplt.xlabel('Study Hours')\nplt.ylabel('Exam Score')\nplt.title('Hours vs Scores')\n\n# Add regression line\nx = df['Hours'].values.reshape(-1, 1)\ny = df['Scores'].values\ncoef = np.polyfit(df['Hours'], df['Scores'], 1)\npoly1d = np.poly1d(coef)\nplt.plot(df['Hours'], poly1d(df['Hours']), 'r--')\n\nplt.show()",
    },
    {
      id: "car-price",
      name: "Car Price Prediction",
      description: "Dataset with various car features and their prices.",
      algorithm: "linear-regression",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load the dataset\ndf = pd.read_csv('car_price.csv')\n\n# Display the first 5 rows\nprint(df.head())\n\n# Correlation with price\ncorrelation = df.corr()['price'].sort_values(ascending=False)\nprint('\\nCorrelation with Price:')\nprint(correlation)\n\n# Visualize top correlations\ntop_features = correlation.index[1:6]  # Exclude price itself\nplt.figure(figsize=(12, 8))\nfor i, feature in enumerate(top_features):\n    plt.subplot(2, 3, i+1)\n    plt.scatter(df[feature], df['price'])\n    plt.xlabel(feature)\n    plt.ylabel('Price')\n    plt.title(f'{feature} vs Price')\n\nplt.tight_layout()\nplt.show()",
    },
  
    // Logistic Regression Datasets
    {
      id: "iris",
      name: "Iris Flower Classification",
      description: "Classic dataset for classification of iris flowers into three species.",
      algorithm: "logistic-regression",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load the dataset\ndf = pd.read_csv('iris.csv')\n\n# Display the first 5 rows\nprint(df.head())\n\n# Basic statistics\nprint(df.describe())\n\n# Visualize the data\nsns.pairplot(df, hue='species')\nplt.suptitle('Iris Dataset - Feature Relationships by Species', y=1.02)\nplt.show()",
    },
    {
      id: "diabetes",
      name: "Diabetes Prediction",
      description: "Dataset for predicting diabetes based on diagnostic measurements.",
      algorithm: "logistic-regression",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load the dataset\ndf = pd.read_csv('diabetes.csv')\n\n# Display the first 5 rows\nprint(df.head())\n\n# Distribution of outcome\nplt.figure(figsize=(8, 6))\nsns.countplot(x='Outcome', data=df)\nplt.title('Distribution of Diabetes Outcome')\nplt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')\nplt.ylabel('Count')\nplt.show()\n\n# Feature distributions by outcome\nplt.figure(figsize=(15, 10))\nfor i, feature in enumerate(df.columns[:-1]):\n    plt.subplot(2, 4, i+1)\n    sns.boxplot(x='Outcome', y=feature, data=df)\n    plt.title(f'{feature} by Outcome')\n\nplt.tight_layout()\nplt.show()",
    },
  
    // K-Nearest Neighbors Datasets
    {
      id: "wine",
      name: "Wine Classification",
      description: "Dataset for classifying wines into three categories based on chemical attributes.",
      algorithm: "k-nearest-neighbors",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\n# Load the dataset\ndf = pd.read_csv('wine.csv')\n\n# Display the first 5 rows\nprint(df.head())\n\n# Basic statistics\nprint(df.describe())\n\n# Standardize the data\nX = df.drop('target', axis=1)\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# Apply PCA for visualization\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_scaled)\n\n# Create a DataFrame with PCA results\npca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])\npca_df['target'] = df['target']\n\n# Visualize in 2D space\nplt.figure(figsize=(10, 8))\nsns.scatterplot(x='PC1', y='PC2', hue='target', data=pca_df, palette='viridis', s=100)\nplt.title('Wine Classification - PCA Visualization')\nplt.show()",
    },
  
    // Random Forest Datasets
    {
      id: "breast-cancer",
      name: "Breast Cancer Detection",
      description: "Dataset for detecting breast cancer from diagnostic features.",
      algorithm: "random-forest",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load the dataset\ndf = pd.read_csv('breast_cancer.csv')\n\n# Display the first 5 rows\nprint(df.head())\n\n# Distribution of diagnosis\nplt.figure(figsize=(8, 6))\nsns.countplot(x='diagnosis', data=df)\nplt.title('Distribution of Diagnosis')\nplt.xlabel('Diagnosis (M = Malignant, B = Benign)')\nplt.ylabel('Count')\nplt.show()\n\n# Feature importance visualization (simulated)\nfeature_importance = {\n    'radius_mean': 0.18,\n    'texture_mean': 0.05,\n    'perimeter_mean': 0.16,\n    'area_mean': 0.12,\n    'smoothness_mean': 0.08,\n    'compactness_mean': 0.09,\n    'concavity_mean': 0.14,\n    'symmetry_mean': 0.06,\n    'fractal_dimension_mean': 0.04\n}\n\nplt.figure(figsize=(12, 6))\nsns.barplot(x=list(feature_importance.keys()), y=list(feature_importance.values()))\nplt.title('Feature Importance (Simulated)')\nplt.xticks(rotation=45, ha='right')\nplt.tight_layout()\nplt.show()",
    },
  
    // Support Vector Machines Datasets
    {
      id: "digits",
      name: "Handwritten Digits",
      description: "Dataset of handwritten digits for classification.",
      algorithm: "support-vector-machines",
      previewCode:
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# Load the dataset (simulated)\ndigits = np.random.rand(64, 64) * 255\n\n# Display a sample digit\nplt.figure(figsize=(6, 6))\nplt.imshow(digits, cmap='gray')\nplt.title('Sample Handwritten Digit')\nplt.axis('off')\nplt.show()\n\n# Display multiple digits (simulated)\nplt.figure(figsize=(10, 8))\nfor i in range(15):\n    plt.subplot(3, 5, i+1)\n    digit = np.random.rand(8, 8) * 255\n    plt.imshow(digit, cmap='gray')\n    plt.title(f'Digit: {np.random.randint(0, 10)}')\n    plt.axis('off')\n\nplt.tight_layout()\nplt.show()",
    },
  ]
  
  export function getDatasetsByAlgorithm(algorithmId: string): Dataset[] {
    return datasets.filter((dataset) => dataset.algorithm === algorithmId)
  }
  
  export function getDatasetById(datasetId: string): Dataset | undefined {
    return datasets.find((dataset) => dataset.id === datasetId)
  }
  
  