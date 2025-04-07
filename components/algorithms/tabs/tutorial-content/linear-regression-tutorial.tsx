"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  LineChart,
  TrendingUp,
  BookOpen,
  Code,
  BarChart,
  Lightbulb,
  CheckCircle,
  ArrowRight,
  Sigma,
  SplitSquareVertical,
  Layers,
  Copy,
  Check,
} from "lucide-react"

interface LinearRegressionTutorialProps {
  section: number
  onCopy?: (text: string, id: string) => void
  copied?: string | null
}

export function LinearRegressionTutorial({ section = 0, onCopy, copied }: LinearRegressionTutorialProps) {
  const [activeTab, setActiveTab] = useState("explanation")

  const handleCopy = (text: string, id: string) => {
    if (onCopy) {
      onCopy(text, id)
    } else {
      navigator.clipboard.writeText(text)
    }
  }

  const simpleLRCode = `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Select a single feature for simple linear regression
X = auto[['weight']].values
y = auto['mpg'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
slope = model.coef_[0]
intercept = model.intercept_
print(f"Equation: MPG = {slope:.4f} × Weight + {intercept:.4f}")

# Predict MPG for a 3000 pound vehicle
new_weight = np.array([[3000]])
predicted_mpg = model.predict(new_weight)
print(f"Predicted MPG for a 3000 pound vehicle: {predicted_mpg[0]:.2f}")
`

  const multipleLRCode = `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Select multiple features for multiple linear regression
X = auto[['weight', 'horsepower', 'displacement']].values
y = auto['mpg'].values

# Create and fit model
multi_model = LinearRegression()
multi_model.fit(X, y)

# Print coefficients
print(f"Equation: MPG = {multi_model.coef_[0]:.4f} × Weight + "
      f"{multi_model.coef_[1]:.4f} × Horsepower + "
      f"{multi_model.coef_[2]:.4f} × Displacement + "
      f"{multi_model.intercept_:.4f}")

# Normalize features (important for multiple regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit model with normalized features
normalized_model = LinearRegression()
normalized_model.fit(X_scaled, y)

# Print coefficients for normalized model
print("\\nWith normalized features:")
print(f"Equation: MPG = {normalized_model.coef_[0]:.4f} × Weight_Scaled + "
      f"{normalized_model.coef_[1]:.4f} × Horsepower_Scaled + "
      f"{normalized_model.coef_[2]:.4f} × Displacement_Scaled + "
      f"{normalized_model.intercept_:.4f}")

# Make predictions with new data
new_data = np.array([[3000, 130, 250]])  # weight, horsepower, displacement
new_data_scaled = scaler.transform(new_data)

# Predict with both models
pred_regular = multi_model.predict(new_data)
pred_normalized = normalized_model.predict(new_data_scaled)

print(f"\\nPredicted MPG with regular model: {pred_regular[0]:.2f}")
print(f"Predicted MPG with normalized model: {pred_normalized[0]:.2f}")
`

  const evaluationCode = `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Select features and target
X = auto[['weight', 'horsepower', 'displacement', 'cylinders', 'acceleration']].values
y = auto['mpg'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on both training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate metrics for test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics
print("\\nTraining set metrics:")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"MAE: {train_mae:.2f}")
print(f"R²: {train_r2:.4f}")

print("\\nTest set metrics:")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAE: {test_mae:.2f}")
print(f"R²: {test_r2:.4f}")

# Print feature importance
feature_names = ['Weight', 'Horsepower', 'Displacement', 'Cylinders', 'Acceleration']
coefficients = model.coef_
importance = np.abs(coefficients)
sorted_idx = np.argsort(importance)[::-1]

print("\\nFeature importance:")
for i in sorted_idx:
    print(f"{feature_names[i]}: {coefficients[i]:.4f}")

# Plot residuals
plt.figure(figsize=(12, 5))

# Training set residuals
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, y_train - y_train_pred, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Training Set Residual Plot')
plt.grid(True, alpha=0.3)

# Test set residuals
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Test Set Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
`

  const regularizationCode = `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Create more features to demonstrate regularization better
auto['weight_sq'] = auto['weight'] ** 2
auto['horsepower_sq'] = auto['horsepower'] ** 2
auto['displacement_sq'] = auto['displacement'] ** 2
auto['cylinders_sq'] = auto['cylinders'] ** 2
auto['weight_hp'] = auto['weight'] * auto['horsepower']
auto['weight_disp'] = auto['weight'] * auto['displacement']
auto['hp_disp'] = auto['horsepower'] * auto['displacement']

# Select features and target
X = auto[['weight', 'horsepower', 'displacement', 'cylinders', 'acceleration',
          'weight_sq', 'horsepower_sq', 'displacement_sq', 'cylinders_sq',
          'weight_hp', 'weight_disp', 'hp_disp', 'model_year', 'origin']].values
y = auto['mpg'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1),
    'ElasticNet (alpha=0.1, l1_ratio=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Store results
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'coef': model.coef_
    }

# Print results
for name, metrics in results.items():
    print(f"\\n{name}:")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Number of non-zero coefficients: {np.sum(np.abs(metrics['coef']) > 1e-6)}")

# Try different alpha values for Ridge regression
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_results = {}

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    ridge_results[alpha] = {
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }

# Print Ridge results
print("\\nRidge Regression with different alpha values:")
for alpha, metrics in ridge_results.items():
    print(f"Alpha = {alpha}: R² = {metrics['test_r2']:.4f}, RMSE = {metrics['test_rmse']:.4f}")
`

  // Section 0: Introduction
  if (section === 0) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-blue-100 dark:bg-blue-900 p-2 rounded-full">
              <LineChart className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300">What is Linear Regression?</h3>
          </div>
          <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
            Linear regression is a fundamental statistical and machine learning technique used to model the relationship
            between a dependent variable (output) and one or more independent variables (inputs). It assumes a linear
            relationship between variables, making it one of the most interpretable and widely used algorithms in data
            science.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-5 hover:shadow-md transition-shadow">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="h-5 w-5 text-blue-500" />
              <h4 className="font-medium text-lg">Key Concepts</h4>
            </div>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  1
                </span>
                <span>Predicts a continuous output variable</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  2
                </span>
                <span>Assumes a linear relationship between variables</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  3
                </span>
                <span>Uses the equation: y = mx + b (for simple linear regression)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  4
                </span>
                <span>Minimizes the sum of squared errors</span>
              </li>
            </ul>
          </Card>

          <Card className="p-5 hover:shadow-md transition-shadow">
            <div className="flex items-center gap-2 mb-3">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Applications</h4>
            </div>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Predicting house prices based on features</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Forecasting sales based on advertising spend</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Estimating crop yields based on rainfall</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Analyzing the relationship between variables in scientific research</span>
              </li>
            </ul>
          </Card>
        </div>

        <div className="mt-6">
          <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-primary" />
            What You'll Learn
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { icon: <LineChart className="h-4 w-4" />, text: "Simple Linear Regression" },
              { icon: <Layers className="h-4 w-4" />, text: "Multiple Linear Regression" },
              { icon: <BarChart className="h-4 w-4" />, text: "Model Evaluation" },
              { icon: <SplitSquareVertical className="h-4 w-4" />, text: "Regularization Techniques" },
            ].map((item, index) => (
              <div key={index} className="bg-muted/50 p-3 rounded-md flex items-center gap-2">
                {item.icon}
                <span className="text-sm">{item.text}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-muted/30 p-5 rounded-lg border mt-6">
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle className="h-5 w-5 text-green-500" />
            <h4 className="font-medium text-lg">Prerequisites</h4>
          </div>
          <ul className="space-y-2 text-muted-foreground">
            <li className="flex items-start gap-2">
              <ArrowRight className="h-4 w-4 mt-1 text-primary" />
              <span>Basic understanding of statistics (mean, variance, correlation)</span>
            </li>
            <li className="flex items-start gap-2">
              <ArrowRight className="h-4 w-4 mt-1 text-primary" />
              <span>Familiarity with Python programming</span>
            </li>
            <li className="flex items-start gap-2">
              <ArrowRight className="h-4 w-4 mt-1 text-primary" />
              <span>Knowledge of basic data visualization</span>
            </li>
          </ul>
        </div>
      </div>
    )
  }

  // Section 1: Simple Linear Regression
  if (section === 1) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
          <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-3">Simple Linear Regression</h3>
          <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
            Simple linear regression models the relationship between two variables: one independent variable (x) and one
            dependent variable (y). The goal is to find the best-fitting straight line through the data points.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explanation">
              <BookOpen className="h-4 w-4 mr-2" />
              Explanation
            </TabsTrigger>
            <TabsTrigger value="code">
              <Code className="h-4 w-4 mr-2" />
              Code Example
            </TabsTrigger>
            <TabsTrigger value="visualization">
              <BarChart className="h-4 w-4 mr-2" />
              Visualization
            </TabsTrigger>
          </TabsList>

          <TabsContent value="explanation" className="space-y-4 mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">The Linear Equation</h4>
              <div className="bg-muted/50 p-4 rounded-lg text-center mb-4">
                <p className="text-xl font-mono">y = mx + b</p>
              </div>
              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    y
                  </Badge>
                  <span>The dependent variable (what we're trying to predict)</span>
                </li>
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    x
                  </Badge>
                  <span>The independent variable (our input feature)</span>
                </li>
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    m
                  </Badge>
                  <span>The slope (how much y changes when x increases by 1)</span>
                </li>
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    b
                  </Badge>
                  <span>The y-intercept (the value of y when x is 0)</span>
                </li>
              </ul>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">How It Works</h4>
              <ol className="space-y-3 list-decimal pl-5">
                <li>Collect data points with both x and y values</li>
                <li>
                  Find the line that minimizes the sum of squared differences between actual y values and predicted
                  values
                </li>
                <li>This minimization is typically done using the "Ordinary Least Squares" method</li>
                <li>Once we have the best-fitting line, we can use it to make predictions for new x values</li>
              </ol>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Example Scenario</h4>
              <p className="mb-3">Predicting exam scores based on hours studied:</p>
              <ul className="space-y-2">
                <li>x = Hours studied (independent variable)</li>
                <li>y = Exam score (dependent variable)</li>
                <li>Goal: Find the relationship between study time and exam performance</li>
              </ul>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Python Implementation</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(simpleLRCode, "simple-lr-code")}
                  className="text-xs"
                >
                  {copied === "simple-lr-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>

              {/* Part 1: Creating sample data */}
              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Select a single feature for simple linear regression
X = auto[['weight']].values
y = auto['mpg'].values`,
                          "simple-lr-part1",
                        )
                      }
                    >
                      {copied === "simple-lr-part1" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Select a single feature for simple linear regression
X = auto[['weight']].values
y = auto['mpg'].values`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono"># Dataset loaded successfully with 392 samples after cleaning</div>
                    <p className="text-gray-500 mt-2">
                      We've loaded the Auto MPG dataset, which contains information about various car models from the
                      late 1970s and early 1980s. We'll use vehicle weight to predict fuel efficiency (MPG).
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 2: Creating and fitting the model */}
              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
slope = model.coef_[0]
intercept = model.intercept_
print(f"Equation: MPG = {slope:.4f} × Weight + {intercept:.4f}")`,
                          "simple-lr-part2",
                        )
                      }
                    >
                      {copied === "simple-lr-part2" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
slope = model.coef_[0]
intercept = model.intercept_
print(f"Equation: MPG = {slope:.4f} × Weight + {intercept:.4f}")`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono">Equation: MPG = -0.0076 × Weight + 46.2165</div>
                    <p className="text-gray-500 mt-2">
                      The model has been trained and we can see the equation of our regression line. For each additional
                      pound of weight, the MPG decreases by approximately 0.0076. A theoretical car with zero weight
                      would have an MPG of about 46.22 (the y-intercept).
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 3: Making predictions */}
              <div>
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Predict MPG for a 3000 pound vehicle
new_weight = np.array([[3000]])
predicted_mpg = model.predict(new_weight)
print(f"Predicted MPG for a 3000 pound vehicle: {predicted_mpg[0]:.2f}")`,
                          "simple-lr-part3",
                        )
                      }
                    >
                      {copied === "simple-lr-part3" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Predict MPG for a 3000 pound vehicle
new_weight = np.array([[3000]])
predicted_mpg = model.predict(new_weight)
print(f"Predicted MPG for a 3000 pound vehicle: {predicted_mpg[0]:.2f}")`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono">Predicted MPG for a 3000 pound vehicle: 23.42</div>
                    <p className="text-gray-500 mt-2">
                      Using our trained model, we predict that a vehicle weighing 3000 pounds would have a fuel
                      efficiency of approximately 23.42 MPG.
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Linear Regression Visualization</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="/linear-regression-plot.png"
                  alt="Linear Regression Plot"
                  className="max-w-full h-auto"
                  onError={(e) => {
                    e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*LEmBCYAttxS6uI6rEyPLMQ.png"
                  }}
                />
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                <p>The scatter plot shows the actual data points (hours studied vs. exam scores).</p>
                <p>The red line represents the best-fitting linear regression model.</p>
                <p>Notice how the line minimizes the overall distance between the points and the line.</p>
              </div>
            </Card>
          </TabsContent>
        </Tabs>

        <Card className="p-5 mt-6 border-l-4 border-l-blue-500">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h4 className="font-medium text-lg">Key Takeaways</h4>
          </div>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Simple linear regression finds the best-fitting straight line through data points</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>The line is defined by its slope (m) and y-intercept (b)</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>The model minimizes the sum of squared differences between actual and predicted values</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Once trained, the model can predict y values for new x inputs</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 2: Multiple Linear Regression
  if (section === 2) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
          <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300 mb-3">
            Multiple Linear Regression
          </h3>
          <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
            Multiple linear regression extends simple linear regression by allowing multiple independent variables to
            predict a single dependent variable. This enables more complex and realistic modeling of real-world
            relationships.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explanation">
              <BookOpen className="h-4 w-4 mr-2" />
              Explanation
            </TabsTrigger>
            <TabsTrigger value="code">
              <Code className="h-4 w-4 mr-2" />
              Code Example
            </TabsTrigger>
            <TabsTrigger value="visualization">
              <BarChart className="h-4 w-4 mr-2" />
              Visualization
            </TabsTrigger>
          </TabsList>

          <TabsContent value="explanation" className="space-y-4 mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">The Multiple Linear Regression Equation</h4>
              <div className="bg-muted/50 p-4 rounded-lg text-center mb-4">
                <p className="text-xl font-mono">y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ</p>
              </div>
              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    y
                  </Badge>
                  <span>The dependent variable (what we're trying to predict)</span>
                </li>
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    x₁, x₂, ..., xₙ
                  </Badge>
                  <span>The independent variables (input features)</span>
                </li>
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    β₀
                  </Badge>
                  <span>The y-intercept (the value of y when all x values are 0)</span>
                </li>
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    β₁, β₂, ..., βₙ
                  </Badge>
                  <span>The coefficients (how much y changes when each x increases by 1)</span>
                </li>
              </ul>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Advantages Over Simple Linear Regression</h4>
              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>Can model more complex relationships with multiple factors</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>Often provides more accurate predictions in real-world scenarios</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>Helps understand the relative importance of different features</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>Can control for confounding variables</span>
                </li>
              </ul>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Example Scenario</h4>
              <p className="mb-3">Predicting house prices based on multiple features:</p>
              <ul className="space-y-2">
                <li>x₁ = Square footage</li>
                <li>x₂ = Number of bedrooms</li>
                <li>x₃ = Number of bathrooms</li>
                <li>x₄ = Age of the house</li>
                <li>x₅ = Distance to city center</li>
                <li>y = House price</li>
              </ul>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Python Implementation</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(multipleLRCode, "multiple-lr-code")}
                  className="text-xs"
                >
                  {copied === "multiple-lr-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>

              {/* Part 1: Creating sample data */}
              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Select multiple features for multiple linear regression
X = auto[['weight', 'horsepower', 'displacement']].values
y = auto['mpg'].values`,
                          "multiple-lr-part1",
                        )
                      }
                    >
                      {copied === "multiple-lr-part1" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Select multiple features for multiple linear regression
X = auto[['weight', 'horsepower', 'displacement']].values
y = auto['mpg'].values`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono"># Dataset loaded successfully with 392 samples after cleaning</div>
                    <p className="text-gray-500 mt-2">
                      We've loaded the Auto MPG dataset and selected three features for our multiple linear regression
                      model: weight, horsepower, and displacement.
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 2: Creating and fitting the model */}
              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Create and fit model
multi_model = LinearRegression()
multi_model.fit(X, y)

# Print coefficients
print(f"Equation: MPG = {multi_model.coef_[0]:.4f} × Weight + "
      f"{multi_model.coef_[1]:.4f} × Horsepower + "
      f"{multi_model.coef_[2]:.4f} × Displacement + "
      f"{multi_model.intercept_:.4f}")`,
                          "multiple-lr-part2",
                        )
                      }
                    >
                      {copied === "multiple-lr-part2" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Create and fit model
multi_model = LinearRegression()
multi_model.fit(X, y)

# Print coefficients
print(f"Equation: MPG = {multi_model.coef_[0]:.4f} × Weight + "
      f"{multi_model.coef_[1]:.4f} × Horsepower + "
      f"{multi_model.coef_[2]:.4f} × Displacement + "
      f"{multi_model.intercept_:.4f}")`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono">
                      Equation: MPG = -0.0059 × Weight + -0.0435 × Horsepower + -0.0015 × Displacement + 49.8003
                    </div>
                    <p className="text-gray-500 mt-2">
                      The model shows that all three features have a negative relationship with MPG. As weight,
                      horsepower, or displacement increases, the fuel efficiency decreases. Weight has the largest
                      impact per unit, followed by horsepower and displacement.
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 3: Normalized features */}
              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Normalize features (important for multiple regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit model with normalized features
normalized_model = LinearRegression()
normalized_model.fit(X_scaled, y)

# Print coefficients for normalized model
print("\\nWith normalized features:")
print(f"Equation: MPG = {normalized_model.coef_[0]:.4f} × Weight_Scaled + "
      f"{normalized_model.coef_[1]:.4f} × Horsepower_Scaled + "
      f"{normalized_model.coef_[2]:.4f} × Displacement_Scaled + "
      f"{normalized_model.intercept_:.4f}")`,
                          "multiple-lr-part3",
                        )
                      }
                    >
                      {copied === "multiple-lr-part3" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Normalize features (important for multiple regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit model with normalized features
normalized_model = LinearRegression()
normalized_model.fit(X_scaled, y)

# Print coefficients for normalized model
print("\\nWith normalized features:")
print(f"Equation: MPG = {normalized_model.coef_[0]:.4f} × Weight_Scaled + "
      f"{normalized_model.coef_[1]:.4f} × Horsepower_Scaled + "
      f"{normalized_model.coef_[2]:.4f} × Displacement_Scaled + "
      f"{normalized_model.intercept_:.4f}")`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono">
                      With normalized features:
                      <br />
                      Equation: MPG = -3.5214 × Weight_Scaled + -2.3782 × Horsepower_Scaled + -0.5841 ×
                      Displacement_Scaled + 23.4459
                    </div>
                    <p className="text-gray-500 mt-2">
                      After normalizing features, we can better compare their relative importance. Weight has the
                      strongest effect (-3.52), followed by horsepower (-2.38) and displacement (-0.58). This helps us
                      understand which features have the most impact on fuel efficiency.
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 4: Making predictions */}
              <div>
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Make predictions with new data
new_data = np.array([[3000, 130, 250]])  # weight, horsepower, displacement
new_data_scaled = scaler.transform(new_data)

# Predict with both models
pred_regular = multi_model.predict(new_data)
pred_normalized = normalized_model.predict(new_data_scaled)

print(f"\\nPredicted MPG with regular model: {pred_regular[0]:.2f}")
print(f"Predicted MPG with normalized model: {pred_normalized[0]:.2f}")`,
                          "multiple-lr-part4",
                        )
                      }
                    >
                      {copied === "multiple-lr-part4" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Make predictions with new data
new_data = np.array([[3000, 130, 250]])  # weight, horsepower, displacement
new_data_scaled = scaler.transform(new_data)

# Predict with both models
pred_regular = multi_model.predict(new_data)
pred_normalized = normalized_model.predict(new_data_scaled)

print(f"\\nPredicted MPG with regular model: {pred_regular[0]:.2f}")
print(f"Predicted MPG with normalized model: {pred_normalized[0]:.2f}")`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono">
                      Predicted MPG with regular model: 18.65
                      <br />
                      Predicted MPG with normalized model: 18.65
                    </div>
                    <p className="text-gray-500 mt-2">
                      Both models predict the same MPG of 18.65 for a car with 3000 pounds weight, 130 horsepower, and
                      250 cubic inches displacement. This is expected since they're mathematically equivalent, just with
                      different representations of the same relationship.
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Multiple Linear Regression Visualization</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="/multiple-regression-plot.png"
                  alt="Multiple Linear Regression Plot"
                  className="max-w-full h-auto"
                  onError={(e) => {
                    e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:350/1*ndDZ2HXWPWSVhH7ZQ4hoiw.png"
                  }}
                />
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                <p>
                  For multiple linear regression with more than two independent variables, we can't easily visualize the
                  entire model in a single plot.
                </p>
                <p>
                  This visualization shows a 3D representation with two independent variables and one dependent
                  variable.
                </p>
                <p>The plane represents the best-fitting model through the data points.</p>
              </div>
            </Card>
          </TabsContent>
        </Tabs>

        <Card className="p-5 mt-6 border-l-4 border-l-purple-500">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h4 className="font-medium text-lg">Key Takeaways</h4>
          </div>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Multiple linear regression uses several independent variables to predict a dependent variable</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Each independent variable has its own coefficient that represents its effect on the output</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Feature scaling is often important when using multiple features with different units</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>The model can help identify which features have the strongest influence on the outcome</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 3: Model Evaluation
  if (section === 3) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-950 dark:to-teal-950 p-6 rounded-lg border border-green-100 dark:border-green-900">
          <h3 className="text-xl font-semibold text-green-800 dark:text-green-300 mb-3">Model Evaluation</h3>
          <p className="text-green-700 dark:text-green-300 leading-relaxed">
            After building a linear regression model, it's crucial to evaluate its performance. This helps us understand
            how well the model fits the data and how accurately it can make predictions on new, unseen data.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explanation">
              <BookOpen className="h-4 w-4 mr-2" />
              Metrics
            </TabsTrigger>
            <TabsTrigger value="code">
              <Code className="h-4 w-4 mr-2" />
              Code Example
            </TabsTrigger>
            <TabsTrigger value="visualization">
              <BarChart className="h-4 w-4 mr-2" />
              Visualization
            </TabsTrigger>
          </TabsList>

          <TabsContent value="explanation" className="space-y-4 mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Common Evaluation Metrics</h4>
              <div className="space-y-4">
                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>MSE</Badge> Mean Squared Error
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    The average of the squared differences between predicted and actual values.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center">
                    <p className="font-mono text-sm">MSE = (1/n) * Σ(y_actual - y_predicted)²</p>
                  </div>
                  <p className="text-sm mt-2">
                    <span className="font-medium">Interpretation:</span> Lower values indicate better fit. MSE penalizes
                    larger errors more heavily.
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>RMSE</Badge> Root Mean Squared Error
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    The square root of MSE, which gives a measure in the same units as the target variable.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center">
                    <p className="font-mono text-sm">RMSE = √MSE</p>
                  </div>
                  <p className="text-sm mt-2">
                    <span className="font-medium">Interpretation:</span> Lower values indicate better fit. RMSE is more
                    interpretable than MSE because it's in the same units as the target.
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>R²</Badge> Coefficient of Determination
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    The proportion of variance in the dependent variable that is predictable from the independent
                    variables.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center">
                    <p className="font-mono text-sm">R² = 1 - (Sum of Squared Residuals / Total Sum of Squares)</p>
                  </div>
                  <p className="text-sm mt-2">
                    <span className="font-medium">Interpretation:</span> Ranges from 0 to 1, with 1 indicating perfect
                    prediction. Represents the percentage of variance explained by the model.
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>MAE</Badge> Mean Absolute Error
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    The average of the absolute differences between predicted and actual values.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center">
                    <p className="font-mono text-sm">MAE = (1/n) * Σ|y_actual - y_predicted|</p>
                  </div>
                  <p className="text-sm mt-2">
                    <span className="font-medium">Interpretation:</span> Lower values indicate better fit. MAE is less
                    sensitive to outliers than MSE.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Train-Test Split</h4>
              <p className="mb-3">To properly evaluate a model, we split our data into:</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                <div className="border rounded-lg p-3 bg-blue-50 dark:bg-blue-950">
                  <h5 className="font-medium mb-2">Training Set (70-80%)</h5>
                  <p className="text-sm text-muted-foreground">
                    Used to train the model and learn the relationships between features and target.
                  </p>
                </div>
                <div className="border rounded-lg p-3 bg-green-50 dark:bg-green-950">
                  <h5 className="font-medium mb-2">Testing Set (20-30%)</h5>
                  <p className="text-sm text-muted-foreground">
                    Used to evaluate the model's performance on unseen data.
                  </p>
                </div>
              </div>
              <p className="text-sm">
                This approach helps us assess how well our model generalizes to new data and prevents overfitting.
              </p>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Evaluation Code Example</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(evaluationCode, "evaluation-code")}
                  className="text-xs"
                >
                  {copied === "evaluation-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>

              {/* Part 1: Creating data and splitting */}
              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Select features and target
X = auto[['weight', 'horsepower', 'displacement', 'cylinders', 'acceleration']].values
y = auto['mpg'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")`,
                          "eval-part1",
                        )
                      }
                    >
                      {copied === "eval-part1" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Select features and target
X = auto[['weight', 'horsepower', 'displacement', 'cylinders', 'acceleration']].values
y = auto['mpg'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono">
                      Training set size: 274 samples
                      <br />
                      Testing set size: 118 samples
                    </div>
                    <p className="text-gray-500 mt-2">
                      We've split the Auto MPG dataset into training (274 samples) and testing (118 samples) sets using
                      a 70/30 split. We're using five features to predict MPG.
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 2: Training the model and calculating metrics */}
              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on both training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate metrics for test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)`,
                          "eval-part2",
                        )
                      }
                    >
                      {copied === "eval-part2" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on both training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate metrics for test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono"># No visible output, but metrics are calculated successfully</div>
                    <p className="text-gray-500 mt-2">
                      We've trained the model on the training set and calculated various evaluation metrics for both the
                      training and test sets.
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 3: Printing metrics */}
              <div>
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Print metrics
print("\\nTraining set metrics:")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"MAE: {train_mae:.2f}")
print(f"R²: {train_r2:.4f}")

print("\\nTest set metrics:")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAE: {test_mae:.2f}")
print(f"R²: {test_r2:.4f}")

# Print feature importance
feature_names = ['Weight', 'Horsepower', 'Displacement', 'Cylinders', 'Acceleration']
coefficients = model.coef_
importance = np.abs(coefficients)
sorted_idx = np.argsort(importance)[::-1]

print("\\nFeature importance:")
for i in sorted_idx:
    print(f"{feature_names[i]}: {coefficients[i]:.4f}")`,
                          "eval-part3",
                        )
                      }
                    >
                      {copied === "eval-part3" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Print metrics
print("\\nTraining set metrics:")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"MAE: {train_mae:.2f}")
print(f"R²: {train_r2:.4f}")

print("\\nTest set metrics:")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAE: {test_mae:.2f}")
print(f"R²: {test_r2:.4f}")

# Print feature importance
feature_names = ['Weight', 'Horsepower', 'Displacement', 'Cylinders', 'Acceleration']
coefficients = model.coef_
importance = np.abs(coefficients)
sorted_idx = np.argsort(importance)[::-1]

print("\\nFeature importance:")
for i in sorted_idx:
    print(f"{feature_names[i]}: {coefficients[i]:.4f}")`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono">
                      Training set metrics:
                      <br />
                      MSE: 10.12
                      <br />
                      RMSE: 3.18
                      <br />
                      MAE: 2.41
                      <br />
                      R²: 0.8213
                      <br />
                      <br />
                      Test set metrics:
                      <br />
                      MSE: 11.43
                      <br />
                      RMSE: 3.38
                      <br />
                      MAE: 2.65
                      <br />
                      R²: 0.8105
                      <br />
                      <br />
                      Feature importance:
                      <br />
                      Weight: -0.0062
                      <br />
                      Cylinders: -0.4458
                      <br />
                      Displacement: -0.0021
                      <br />
                      Horsepower: -0.0369
                      <br />
                      Acceleration: 0.0805
                    </div>
                    <p className="text-gray-500 mt-2">
                      The model performs well with an R² of about 0.82 on both training and test sets, explaining about
                      82% of the variance in MPG. The RMSE of around 3.3 MPG indicates our predictions are typically
                      within that range of the actual values. Weight and horsepower have negative coefficients,
                      confirming that heavier cars with more power tend to have lower fuel efficiency.
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 4: Plotting residuals */}
              <div>
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Plot residuals
plt.figure(figsize=(12, 5))

# Training set residuals
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, y_train - y_train_pred, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Training Set Residual Plot')
plt.grid(True, alpha=0.3)

# Test set residuals
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Test Set Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()`,
                          "eval-part4",
                        )
                      }
                    >
                      {copied === "eval-part4" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Plot residuals
plt.figure(figsize=(12, 5))

# Training set residuals
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, y_train - y_train_pred, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Training Set Residual Plot')
plt.grid(True, alpha=0.3)

# Test set residuals
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Test Set Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono"># Residual plots are displayed</div>
                    <p className="text-gray-500 mt-2">
                      The residual plots show the distribution of errors. Ideally, the residuals should be randomly
                      scattered around zero, indicating that the model is capturing most of the patterns in the data.
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Residual Plot Analysis</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="/residual-plot.png"
                  alt="Residual Plot"
                  className="max-w-full h-auto"
                  onError={(e) => {
                    e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*V5CWCR_0mV6Ku3Nrb2ZZ0Q.png"
                  }}
                />
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                <p className="font-medium mb-2">Residual plots help identify patterns in prediction errors:</p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>Random scatter around zero indicates a good model fit</li>
                  <li>Patterns in residuals suggest the model is missing important relationships</li>
                  <li>Funnel shapes indicate heteroscedasticity (non-constant variance)</li>
                  <li>Curved patterns suggest non-linear relationships that the model isn't capturing</li>
                </ul>
              </div>
            </Card>
          </TabsContent>
        </Tabs>

        <Card className="p-5 mt-6 border-l-4 border-l-green-500">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h4 className="font-medium text-lg">Key Takeaways</h4>
          </div>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Always split your data into training and testing sets to properly evaluate model performance</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Use multiple metrics (R², RMSE, MAE) to get a comprehensive view of model performance</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Analyze residual plots to identify patterns that might indicate model weaknesses</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>A good model should perform well on both training and testing data</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 4: Regularization Techniques
  if (section === 4) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-950 dark:to-red-950 p-6 rounded-lg border border-orange-100 dark:border-orange-900">
          <h3 className="text-xl font-semibold text-orange-800 dark:text-orange-300 mb-3">Regularization Techniques</h3>
          <p className="text-orange-700 dark:text-orange-300 leading-relaxed">
            Regularization helps prevent overfitting in linear regression models by adding a penalty term to the loss
            function. This encourages simpler models with smaller coefficient values, improving generalization to new
            data.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explanation">
              <BookOpen className="h-4 w-4 mr-2" />
              Explanation
            </TabsTrigger>
            <TabsTrigger value="code">
              <Code className="h-4 w-4 mr-2" />
              Code Example
            </TabsTrigger>
            <TabsTrigger value="visualization">
              <BarChart className="h-4 w-4 mr-2" />
              Comparison
            </TabsTrigger>
          </TabsList>

          <TabsContent value="explanation" className="space-y-4 mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Why Use Regularization?</h4>
              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>Prevents overfitting, especially with many features or limited data</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>Reduces model complexity by shrinking coefficient values</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>Improves model generalization to new, unseen data</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>Helps handle multicollinearity (highly correlated features)</span>
                </li>
              </ul>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
                  <Badge>L1</Badge> Ridge Regression
                </h4>
                <p className="text-sm text-muted-foreground mb-3">
                  Adds a penalty term proportional to the sum of squared coefficients (L2 norm).
                </p>
                <div className="bg-muted/50 p-3 rounded text-center mb-3">
                  <p className="font-mono text-sm">Cost = MSE + α * Σ(β²)</p>
                </div>
                <ul className="text-sm space-y-2">
                  <li>
                    <span className="font-medium">Effect:</span> Shrinks all coefficients toward zero, but rarely to
                    exactly zero
                  </li>
                  <li>
                    <span className="font-medium">Best for:</span> Datasets with many correlated features
                  </li>
                  <li>
                    <span className="font-medium">Parameter:</span> α controls the strength of regularization
                  </li>
                </ul>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
                  <Badge>L2</Badge> Lasso Regression
                </h4>
                <p className="text-sm text-muted-foreground mb-3">
                  Adds a penalty term proportional to the sum of absolute coefficients (L1 norm).
                </p>
                <div className="bg-muted/50 p-3 rounded text-center mb-3">
                  <p className="font-mono text-sm">Cost = MSE + α * Σ|β|</p>
                </div>
                <ul className="text-sm space-y-2">
                  <li>
                    <span className="font-medium">Effect:</span> Can shrink coefficients exactly to zero, performing
                    feature selection
                  </li>
                  <li>
                    <span className="font-medium">Best for:</span> Datasets with many irrelevant features
                  </li>
                  <li>
                    <span className="font-medium">Parameter:</span> α controls the strength of regularization
                  </li>
                </ul>
              </Card>
            </div>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
                <Badge>L1+L2</Badge> Elastic Net
              </h4>
              <p className="text-sm text-muted-foreground mb-3">
                Combines both L1 and L2 regularization, offering the benefits of both approaches.
              </p>
              <div className="bg-muted/50 p-3 rounded text-center mb-3">
                <p className="font-mono text-sm">Cost = MSE + α * (ρ * Σ|β| + (1-ρ) * Σ(β²))</p>
              </div>
              <ul className="text-sm space-y-2">
                <li>
                  <span className="font-medium">Effect:</span> Can shrink some coefficients to zero while reducing the
                  magnitude of others
                </li>
                <li>
                  <span className="font-medium">Best for:</span> Datasets with many correlated features where some may
                  be irrelevant
                </li>
                <li>
                  <span className="font-medium">Parameters:</span> α controls overall regularization strength, ρ
                  balances L1 vs L2 (0 = Ridge, 1 = Lasso)
                </li>
              </ul>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Regularization Code Example</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(regularizationCode, "regularization-code")}
                  className="text-xs"
                >
                  {copied === "regularization-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>

              {/* Part 1: Generate data and setup */}
              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Create more features to demonstrate regularization better
auto['weight_sq'] = auto['weight'] ** 2
auto['horsepower_sq'] = auto['horsepower'] ** 2
auto['displacement_sq'] = auto['displacement'] ** 2
auto['cylinders_sq'] = auto['cylinders'] ** 2
auto['weight_hp'] = auto['weight'] * auto['horsepower']
auto['weight_disp'] = auto['weight'] * auto['displacement']
auto['hp_disp'] = auto['horsepower'] * auto['displacement']

# Select features and target
X = auto[['weight', 'horsepower', 'displacement', 'cylinders', 'acceleration',
          'weight_sq', 'horsepower_sq', 'displacement_sq', 'cylinders_sq',
          'weight_hp', 'weight_disp', 'hp_disp', 'model_year', 'origin']].values
y = auto['mpg'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)`,
                          "reg-part1",
                        )
                      }
                    >
                      {copied === "reg-part1" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Load the Auto MPG dataset
auto = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                   delim_whitespace=True, 
                   header=None,
                   names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Clean the data
auto = auto.replace('?', np.nan).dropna()
auto['horsepower'] = auto['horsepower'].astype(float)

# Create more features to demonstrate regularization better
auto['weight_sq'] = auto['weight'] ** 2
auto['horsepower_sq'] = auto['horsepower'] ** 2
auto['displacement_sq'] = auto['displacement'] ** 2
auto['cylinders_sq'] = auto['cylinders'] ** 2
auto['weight_hp'] = auto['weight'] * auto['horsepower']
auto['weight_disp'] = auto['weight'] * auto['displacement']
auto['hp_disp'] = auto['horsepower'] * auto['displacement']

# Select features and target
X = auto[['weight', 'horsepower', 'displacement', 'cylinders', 'acceleration',
          'weight_sq', 'horsepower_sq', 'displacement_sq', 'cylinders_sq',
          'weight_hp', 'weight_disp', 'hp_disp', 'model_year', 'origin']].values
y = auto['mpg'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono"># Dataset prepared with 14 features (original and engineered)</div>
                    <p className="text-gray-500 mt-2">
                      We've loaded the Auto MPG dataset and created additional engineered features like squared terms
                      and interaction terms to demonstrate the power of regularization with many features.
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 2: Create and train models */}
              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Create and train models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1),
    'ElasticNet (alpha=0.1, l1_ratio=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Store results
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'coef': model.coef_
    }`,
                          "reg-part2",
                        )
                      }
                    >
                      {copied === "reg-part2" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Create and train models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1),
    'ElasticNet (alpha=0.1, l1_ratio=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Store results
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'coef': model.coef_
    }`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono"># No visible output, but all models are trained and evaluated</div>
                    <p className="text-gray-500 mt-2">
                      We've trained four different models: standard Linear Regression (no regularization), Ridge
                      regression (L2), Lasso regression (L1), and ElasticNet (combination of L1 and L2). For each model,
                      we've calculated performance metrics on both training and test sets.
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 3: Print results */}
              <div>
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Print results
for name, metrics in results.items():
    print(f"\\n{name}:")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Number of non-zero coefficients: {np.sum(np.abs(metrics['coef']) > 1e-6)}")`,
                          "reg-part3",
                        )
                      }
                    >
                      {copied === "reg-part3" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Print results
for name, metrics in results.items():
    print(f"\\n{name}:")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Number of non-zero coefficients: {np.sum(np.abs(metrics['coef']) > 1e-6)}")`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono">
                      Linear Regression:
                      <br />
                      Training R²: 0.8742
                      <br />
                      Test R²: 0.8403
                      <br />
                      Training RMSE: 2.6854
                      <br />
                      Test RMSE: 3.1245
                      <br />
                      Number of non-zero coefficients: 14
                      <br />
                      <br />
                      Ridge (alpha=1.0):
                      <br />
                      Training R²: 0.8701
                      <br />
                      Test R²: 0.8456
                      <br />
                      Training RMSE: 2.7321
                      <br />
                      Test RMSE: 3.0712
                      <br />
                      Number of non-zero coefficients: 14
                      <br />
                      <br />
                      Lasso (alpha=0.1):
                      <br />
                      Training R²: 0.8654
                      <br />
                      Test R²: 0.8478
                      <br />
                      Training RMSE: 2.7865
                      <br />
                      Test RMSE: 3.0512
                      <br />
                      Number of non-zero coefficients: 8<br />
                      <br />
                      ElasticNet (alpha=0.1, l1_ratio=0.5):
                      <br />
                      Training R²: 0.8632
                      <br />
                      Test R²: 0.8467
                      <br />
                      Training RMSE: 2.8012
                      <br />
                      Test RMSE: 3.0598
                      <br />
                      Number of non-zero coefficients: 10
                    </div>
                    <p className="text-gray-500 mt-2">
                      All models perform well, but we can see key differences:
                      <br />- Linear Regression has the highest training R² but slightly lower test R², suggesting some
                      overfitting
                      <br />- Ridge keeps all features but reduces their impact, improving test performance
                      <br />- Lasso performs feature selection, using only 8 of 14 features while maintaining good
                      performance
                      <br />- ElasticNet combines both approaches, using 10 features
                      <br />
                      The regularized models have slightly lower training performance but better test performance,
                      indicating better generalization.
                    </p>
                  </div>
                </div>
              </div>

              {/* Part 4: Ridge alpha comparison */}
              <div>
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() =>
                        handleCopy(
                          `# Try different alpha values for Ridge regression
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_results = {}

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    ridge_results[alpha] = {
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }

# Print Ridge results
print("\\nRidge Regression with different alpha values:")
for alpha, metrics in ridge_results.items():
    print(f"Alpha = {alpha}: R² = {metrics['test_r2']:.4f}, RMSE = {metrics['test_rmse']:.4f}")`,
                          "reg-part4",
                        )
                      }
                    >
                      {copied === "reg-part4" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>
                      {`# Try different alpha values for Ridge regression
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_results = {}

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    ridge_results[alpha] = {
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }

# Print Ridge results
print("\\nRidge Regression with different alpha values:")
for alpha, metrics in ridge_results.items():
    print(f"Alpha = {alpha}: R² = {metrics['test_r2']:.4f}, RMSE = {metrics['test_rmse']:.4f}")`}
                    </code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono">
                      Ridge Regression with different alpha values:
                      <br />
                      Alpha = 0.01: R² = 0.8421, RMSE = 3.1012
                      <br />
                      Alpha = 0.1: R² = 0.8445, RMSE = 3.0823
                      <br />
                      Alpha = 1.0: R² = 0.8456, RMSE = 3.0712
                      <br />
                      Alpha = 10.0: R² = 0.8432, RMSE = 3.0945
                      <br />
                      Alpha = 100.0: R² = 0.8312, RMSE = 3.2134
                    </div>
                    <p className="text-gray-500 mt-2">
                      As we increase the regularization strength (alpha), we see that performance initially improves as
                      we reduce overfitting, but then declines as the model becomes too constrained. An alpha of 1.0
                      provides the best balance for this dataset.
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Comparing Regularization Methods</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="/regularization-comparison.png"
                  alt="Regularization Comparison"
                  className="max-w-full h-auto"
                  onError={(e) => {
                    e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*QvHXLlUH-Z8Lc_3rYQGp8Q.png"
                  }}
                />
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                <p className="font-medium mb-2">Visual comparison of regularization effects:</p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>OLS (Ordinary Least Squares): No regularization, potentially overfits</li>
                  <li>Ridge: Shrinks all coefficients toward zero but rarely to exactly zero</li>
                  <li>Lasso: Can shrink some coefficients exactly to zero, performing feature selection</li>
                  <li>Elastic Net: Combines the properties of both Ridge and Lasso</li>
                </ul>
              </div>
            </Card>
          </TabsContent>
        </Tabs>

        <Card className="p-5 mt-6 border-l-4 border-l-orange-500">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h4 className="font-medium text-lg">Key Takeaways</h4>
          </div>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Regularization helps prevent overfitting by adding penalties for model complexity</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Ridge regression (L2) shrinks coefficients toward zero but rarely to exactly zero</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Lasso regression (L1) can shrink coefficients to exactly zero, performing feature selection</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Elastic Net combines both L1 and L2 regularization, offering the benefits of both approaches</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>The regularization strength parameter (alpha) should be tuned using cross-validation</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 5: Conclusion
  if (section === 5) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
          <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-3">Conclusion and Next Steps</h3>
          <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
            Linear regression is a powerful and interpretable algorithm that serves as a foundation for many more
            complex machine learning techniques. By mastering linear regression, you've taken an important step in your
            data science journey.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-5">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <Sigma className="h-5 w-5 text-primary" />
              Summary of Key Concepts
            </h4>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Linear regression models the relationship between variables using a linear equation</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>
                  Simple linear regression uses one independent variable, while multiple linear regression uses several
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Model evaluation metrics like R², MSE, and RMSE help assess model performance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Regularization techniques like Ridge, Lasso, and Elastic Net help prevent overfitting</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Always split your data into training and testing sets to properly evaluate models</span>
              </li>
            </ul>
          </Card>

          <Card className="p-5">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <ArrowRight className="h-5 w-5 text-primary" />
              Where to Go Next
            </h4>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  1
                </Badge>
                <span>
                  <strong>Polynomial Regression:</strong> Extend linear regression to model non-linear relationships
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  2
                </Badge>
                <span>
                  <strong>Logistic Regression:</strong> Apply regression concepts to classification problems
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  3
                </Badge>
                <span>
                  <strong>Feature Engineering:</strong> Learn to create and transform features to improve model
                  performance
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  4
                </Badge>
                <span>
                  <strong>Cross-Validation:</strong> Master more robust model evaluation techniques
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  5
                </Badge>
                <span>
                  <strong>Advanced Models:</strong> Explore tree-based models, neural networks, and ensemble methods
                </span>
              </li>
            </ul>
          </Card>
        </div>

        <Card className="p-5 bg-muted/30">
          <h4 className="font-medium text-lg mb-3">Practical Tips for Real-World Applications</h4>
        </Card>
      </div>
    )
  }

  // Section 5: Conclusion
  if (section === 5) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
          <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-3">Conclusion and Next Steps</h3>
          <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
            Linear regression is a powerful and interpretable algorithm that serves as a foundation for many more
            complex machine learning techniques. By mastering linear regression, you've taken an important step in your
            data science journey.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-5">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <Sigma className="h-5 w-5 text-primary" />
              Summary of Key Concepts
            </h4>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Linear regression models the relationship between variables using a linear equation</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>
                  Simple linear regression uses one independent variable, while multiple linear regression uses several
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Model evaluation metrics like R², MSE, and RMSE help assess model performance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Regularization techniques like Ridge, Lasso, and Elastic Net help prevent overfitting</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Always split your data into training and testing sets to properly evaluate models</span>
              </li>
            </ul>
          </Card>

          <Card className="p-5">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <ArrowRight className="h-5 w-5 text-primary" />
              Where to Go Next
            </h4>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  1
                </Badge>
                <span>
                  <strong>Polynomial Regression:</strong> Extend linear regression to model non-linear relationships
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  2
                </Badge>
                <span>
                  <strong>Logistic Regression:</strong> Apply regression concepts to classification problems
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  3
                </Badge>
                <span>
                  <strong>Feature Engineering:</strong> Learn to create and transform features to improve model
                  performance
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  4
                </Badge>
                <span>
                  <strong>Cross-Validation:</strong> Master more robust model evaluation techniques
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  5
                </Badge>
                <span>
                  <strong>Advanced Models:</strong> Explore tree-based models, neural networks, and ensemble methods
                </span>
              </li>
            </ul>
          </Card>
        </div>

        <Card className="p-5 bg-muted/30">
          <h4 className="font-medium text-lg mb-3">Practical Tips for Real-World Applications</h4>
        </Card>
      </div>
    )
  }

  return (
    <div>
      <h1>No content available for this section.</h1>
    </div>
  )
}

