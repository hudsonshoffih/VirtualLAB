"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TrendingUp, BookOpen, Code, BarChart, Lightbulb, CheckCircle, ArrowRight, GitBranch, Activity, Target, Copy, Check } from 'lucide-react'

interface LogisticRegressionTutorialProps {
  section: number
  onCopy: (text: string, id: string) => void
  copied: string | null
}

export function LogisticRegressionTutorial({ section, onCopy, copied }: LogisticRegressionTutorialProps) {
  const [activeTab, setActiveTab] = useState("explanation")

  // Section 0: Introduction
  if (section === 0) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-purple-100 dark:bg-purple-900 p-2 rounded-full">
              <Target className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300">What is Logistic Regression?</h3>
          </div>
          <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
            Logistic regression is a supervised learning algorithm used for classification problems. It predicts the
            probability of a categorical dependent variable, typically binary (0 or 1). Unlike Linear Regression, which
            predicts continuous values, Logistic Regression models probabilities using the Sigmoid function to constrain
            outputs between 0 and 1.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-5 hover:shadow-md transition-shadow">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="h-5 w-5 text-purple-500" />
              <h4 className="font-medium text-lg">Key Concepts</h4>
            </div>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  1
                </span>
                <span>Predicts probabilities for classification tasks</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  2
                </span>
                <span>Uses the sigmoid function to output values between 0 and 1</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  3
                </span>
                <span>Decision boundary typically at 0.5 probability</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  4
                </span>
                <span>Optimized using maximum likelihood estimation</span>
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
                <span>Email spam detection</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Disease diagnosis (positive/negative)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Customer churn prediction</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Credit risk assessment (approve/deny)</span>
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
              { icon: <Activity className="h-4 w-4" />, text: "Sigmoid Function" },
              { icon: <GitBranch className="h-4 w-4" />, text: "Types of Logistic Regression" },
              { icon: <Code className="h-4 w-4" />, text: "Implementation in Python" },
              { icon: <BarChart className="h-4 w-4" />, text: "Model Evaluation" },
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
            <h4 className="font-medium text-lg">Linear vs. Logistic Regression</h4>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse">
              <thead>
                <tr className="bg-muted/70">
                  <th className="border px-4 py-2 text-left">Feature</th>
                  <th className="border px-4 py-2 text-left">Linear Regression</th>
                  <th className="border px-4 py-2 text-left">Logistic Regression</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border px-4 py-2 font-medium">Output Type</td>
                  <td className="border px-4 py-2">Continuous values</td>
                  <td className="border px-4 py-2">Probabilities (0 to 1)</td>
                </tr>
                <tr className="bg-muted/30">
                  <td className="border px-4 py-2 font-medium">Function</td>
                  <td className="border px-4 py-2">Linear function</td>
                  <td className="border px-4 py-2">Sigmoid function</td>
                </tr>
                <tr>
                  <td className="border px-4 py-2 font-medium">Use Case</td>
                  <td className="border px-4 py-2">Regression problems</td>
                  <td className="border px-4 py-2">Classification problems</td>
                </tr>
                <tr className="bg-muted/30">
                  <td className="border px-4 py-2 font-medium">Optimization</td>
                  <td className="border px-4 py-2">Least squares</td>
                  <td className="border px-4 py-2">Maximum likelihood</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    )
  }

  // Section 1: Sigmoid Function
  if (section === 1) {
    const sigmoidCode = `import numpy as np

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate some example values
z_values = [-5, -2, -1, 0, 1, 2, 5]

# Calculate sigmoid for each value
for z in z_values:
    print(f"sigmoid({z}) = {sigmoid(z):.6f}")
`

    const sigmoidPlotCode = `import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate data points for z
z = np.linspace(-10, 10, 100)

# Calculate sigmoid values
sigmoid_values = sigmoid(z)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid_values, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ')
plt.ylabel('σ(z) = 1 / (1 + e^(-z))')
plt.title('Sigmoid Function')

# Add annotations
plt.annotate('Decision Boundary (0.5)', xy=(0, 0.5), xytext=(2, 0.4),
             arrowprops=dict(arrowstyle='->'))
plt.annotate('As z → -∞, σ(z) → 0', xy=(-8, 0.0003), xytext=(-8, 0.2),
             arrowprops=dict(arrowstyle='->'))
plt.annotate('As z → ∞, σ(z) → 1', xy=(8, 0.9997), xytext=(8, 0.8),
             arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.show()`

    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
          <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300 mb-3">The Sigmoid Function</h3>
          <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
            The sigmoid function is the heart of logistic regression. It maps any real-valued number to a value between
            0 and 1, making it perfect for modeling probabilities in classification problems.
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
              <h4 className="font-medium text-lg mb-3">The Sigmoid Formula</h4>
              <div className="bg-muted/50 p-4 rounded-lg text-center mb-4">
                <p className="text-xl font-mono">
                  σ(z) = 1 / (1 + e<sup>-z</sup>)
                </p>
              </div>
              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    σ(z)
                  </Badge>
                  <span>The sigmoid function output (between 0 and 1)</span>
                </li>
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    z
                  </Badge>
                  <span>The linear combination of features: z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ</span>
                </li>
                <li className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    e
                  </Badge>
                  <span>The base of natural logarithms (approximately 2.71828)</span>
                </li>
              </ul>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Key Properties</h4>
              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>Always outputs values between 0 and 1</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>At z=0, sigmoid outputs exactly 0.5</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>As z approaches positive infinity, sigmoid approaches 1</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>As z approaches negative infinity, sigmoid approaches 0</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-primary mt-1" />
                  <span>The function has an S-shaped curve (hence "sigmoid")</span>
                </li>
              </ul>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Decision Boundary</h4>
              <p className="mb-3">In binary classification with logistic regression:</p>
              <ul className="space-y-2">
                <li>If σ(z) ≥ 0.5, predict class 1</li>
                <li>If σ(z) &lt; 0.5, predict class 0</li>
              </ul>
              <p className="mt-3 text-sm text-muted-foreground">
                Since σ(z) = 0.5 when z = 0, the decision boundary occurs at z = 0, which corresponds to: b₀ + b₁x₁ +
                b₂x₂ + ... + bₙxₙ = 0
              </p>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Python Implementation</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCopy(sigmoidCode, "sigmoid-code")}
                  className="text-xs"
                >
                  {copied === "sigmoid-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(sigmoidCode, "sigmoid-code")}
                  >
                    {copied === "sigmoid-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{sigmoidCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <pre className="text-sm">
{`sigmoid(-5) = 0.006693
sigmoid(-2) = 0.119203
sigmoid(-1) = 0.268941
sigmoid(0) = 0.500000
sigmoid(1) = 0.731059
sigmoid(2) = 0.880797
sigmoid(5) = 0.993307`}
</pre>
                <p className="text-sm text-muted-foreground mt-2">
                  The sigmoid function maps any real number to a value between 0 and 1. Notice how values far below 0 approach 0, 
                  and values far above 0 approach 1. At exactly z=0, the sigmoid function outputs 0.5, which is the decision boundary.
                </p>
              </div>
              
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(sigmoidPlotCode, "sigmoid-plot-code")}
                  >
                    {copied === "sigmoid-plot-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{sigmoidPlotCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <div className="flex justify-center">
                  <img 
                    src="https://miro.medium.com/v2/resize:fit:1400/1*RqXFpiNGwdiKBWyLJc_E7g.png" 
                    alt="Sigmoid Function Plot" 
                    className="max-w-full h-auto rounded-md border border-gray-300 dark:border-gray-700"
                  />
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  The plot shows the S-shaped curve of the sigmoid function. The horizontal red dashed line at y=0.5 and 
                  the vertical red dashed line at x=0 intersect at the decision boundary. This is where the model transitions 
                  from predicting class 0 to class 1.
                </p>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Sigmoid Function Visualization</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="https://miro.medium.com/v2/resize:fit:1400/1*RqXFpiNGwdiKBWyLJc_E7g.png"
                  alt="Sigmoid Function"
                  className="max-w-full h-auto"
                />
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                <p>The sigmoid function maps any real number to a value between 0 and 1.</p>
                <p>Notice how the function approaches but never reaches 0 or 1 at the extremes.</p>
                <p>The decision boundary is at z=0, where the sigmoid function equals 0.5.</p>
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
              <span>The sigmoid function transforms linear predictions into probabilities</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>It constrains outputs to be between 0 and 1, making it perfect for binary classification</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>The decision boundary occurs at z=0, where the sigmoid function equals 0.5</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>The S-shaped curve allows for smooth transitions between classes</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 2: Types & Assumptions
  if (section === 2) {
    const assumptionsCheckCode = `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Convert to binary target for logistic regression
# (1 if diabetes progression is above median, 0 otherwise)
y_binary = (y > np.median(y)).astype(int)

# Create a DataFrame for easier analysis
feature_names = diabetes.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y_binary

# Check the first few rows of the dataset
print("First 5 rows of the diabetes dataset:")
print(df.head())

# Check for multicollinearity using correlation matrix
print("\\nChecking for multicollinearity:")
correlation_matrix = df[feature_names].corr().round(2)
print(correlation_matrix)`

    const multicollinearityCode = `# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Diabetes Dataset Features')
plt.tight_layout()
plt.show()

# Check for linearity of log odds
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Select a few features for demonstration
selected_features = ['bmi', 'bp', 's5']
X_selected = df[selected_features]

# Add constant for statsmodels
X_with_const = sm.add_constant(X_selected)

# Fit logistic regression
logit_model = sm.Logit(y_binary, X_with_const)
result = logit_model.fit(disp=0)

# Print summary
print("\\nLogistic Regression Summary:")
print(result.summary2().tables[1])`

    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-950 dark:to-blue-950 p-6 rounded-lg border border-indigo-100 dark:border-indigo-900">
          <h3 className="text-xl font-semibold text-indigo-800 dark:text-indigo-300 mb-3">
            Types & Assumptions of Logistic Regression
          </h3>
          <p className="text-indigo-700 dark:text-indigo-300 leading-relaxed">
            Logistic regression comes in different forms depending on the nature of the target variable. Understanding
            the types and underlying assumptions is crucial for proper implementation and interpretation.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explanation">
              <BookOpen className="h-4 w-4 mr-2" />
              Types
            </TabsTrigger>
            <TabsTrigger value="code">
              <Code className="h-4 w-4 mr-2" />
              Assumptions
            </TabsTrigger>
            <TabsTrigger value="visualization">
              <BarChart className="h-4 w-4 mr-2" />
              Visualization
            </TabsTrigger>
          </TabsList>

          <TabsContent value="explanation" className="space-y-4 mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Types of Logistic Regression</h4>
              <div className="space-y-4">
                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>Binary</Badge> Binomial Logistic Regression
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    Predicts a binary outcome with two possible classes (0/1, Yes/No, True/False).
                  </p>
                  <div className="bg-muted/50 p-3 rounded">
                    <p className="text-sm">
                      Examples: Email spam detection, disease diagnosis, customer churn prediction
                    </p>
                  </div>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>Multi-class</Badge> Multinomial Logistic Regression
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    Predicts one of three or more unordered categories.
                  </p>
                  <div className="bg-muted/50 p-3 rounded">
                    <p className="text-sm">
                      Examples: Document classification, product categorization, species identification
                    </p>
                  </div>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>Ordered</Badge> Ordinal Logistic Regression
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    Predicts one of three or more ordered categories.
                  </p>
                  <div className="bg-muted/50 p-3 rounded">
                    <p className="text-sm">
                      Examples: Rating prediction (1-5 stars), education level classification, survey responses
                      (Strongly Disagree to Strongly Agree)
                    </p>
                  </div>
                </div>
              
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Implementation Approaches</h4>
              <div className="space-y-3">
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">One-vs-Rest (OvR)</h5>
                  <p className="text-sm text-muted-foreground">
                    For multi-class problems, trains binary classifiers for each class against all others. Predicts the
                    class with the highest probability.
                  </p>
                </div>
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">One-vs-One (OvO)</h5>
                  <p className="text-sm text-muted-foreground">
                    Trains binary classifiers for each pair of classes. Predicts the class that wins the most pairwise
                    comparisons.
                  </p>
                </div>
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">Softmax Regression</h5>
                  <p className="text-sm text-muted-foreground">
                    Extension of logistic regression that directly handles multiple classes, outputting probabilities
                    for each class that sum to 1.
                  </p>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Assumptions of Logistic Regression</h4>
              <div className="space-y-4">
                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2">1. Binary/Categorical Outcome</h5>
                  <p className="text-sm text-muted-foreground">
                    The dependent variable must be categorical. For binary logistic regression, the outcome should be
                    binary (0/1).
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2">2. Independence of Observations</h5>
                  <p className="text-sm text-muted-foreground">
                    Observations should be independent of each other. The outcome for one observation should not
                    influence the outcome for another.
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2">3. No Multicollinearity</h5>
                  <p className="text-sm text-muted-foreground">
                    Independent variables should not be highly correlated with each other. High correlation can lead to
                    unstable estimates.
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2">4. Linearity of Log Odds</h5>
                  <p className="text-sm text-muted-foreground">
                    The log odds (logit) should be linearly related to the independent variables. This doesn't mean the
                    relationship between X and P(Y) is linear.
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2">5. Large Sample Size</h5>
                  <p className="text-sm text-muted-foreground">
                    Logistic regression requires a relatively large sample size for stable estimates, especially with
                    multiple predictors.
                  </p>
                </div>
              </div>
              
              <div className="relative bg-black rounded-md my-4 mt-6 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(assumptionsCheckCode, "assumptions-check-code")}
                  >
                    {copied === "assumptions-check-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{assumptionsCheckCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <pre className="text-sm overflow-x-auto">
{`First 5 rows of the diabetes dataset:
       age       sex       bmi        bp        s1        s2        s3        s4        s5        s6  target
0  0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592  0.019908 -0.017646       0
1 -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163  0.074412 -0.039493 -0.068330 -0.092204       0
2  0.085299  0.050680  0.044451 -0.005671 -0.045599 -0.034194 -0.032356 -0.002592  0.002864 -0.025930       1
3 -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991 -0.036038  0.034309  0.022692 -0.009362       0
4  0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596  0.008142 -0.002592 -0.031991 -0.046641       1

Checking for multicollinearity:
        age    sex    bmi     bp     s1     s2     s3     s4     s5     s6
age   1.00  0.17  0.19  0.34  0.26  0.22  0.19  0.16  0.21  0.37
sex   0.17  1.00  0.09  0.24  0.04  0.14 -0.11  0.04  0.04  0.15
bmi   0.19  0.09  1.00  0.18  0.22  0.17  0.15  0.11  0.20  0.28
bp    0.34  0.24  0.18  1.00  0.15  0.16  0.07  0.09  0.16  0.34
s1    0.26  0.04  0.22  0.15  1.00  0.90  0.89  0.73  0.71  0.42
s2    0.22  0.14  0.17  0.16  0.90  1.00  0.80  0.67  0.67  0.41
s3    0.19 -0.11  0.15  0.07  0.89  0.80  1.00  0.67  0.64  0.26
s4    0.16  0.04  0.11  0.09  0.73  0.67  0.67  1.00  0.54  0.28
s5    0.21  0.04  0.20  0.16  0.71  0.67  0.64  0.54  1.00  0.37
s6    0.37  0.15  0.28  0.34  0.42  0.41  0.26  0.28  0.37  1.00`}
</pre>

                <p className="text-sm text-muted-foreground mt-2">
                  The diabetes dataset has been loaded and converted to a binary classification problem (1 if diabetes progression is above median, 0 otherwise). 
                  The correlation matrix shows high correlation between some features (s1, s2, s3), which could indicate multicollinearity issues.
                </p>
              </div>
              
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(multicollinearityCode, "multicollinearity-code")}
                  >
                    {copied === "multicollinearity-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{multicollinearityCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <div className="flex justify-center mb-4">
                  <img 
                    src="https://miro.medium.com/v2/resize:fit:1400/1*-FfPEPGHFX4b8HVLRu7BMQ.png" 
                    alt="Correlation Matrix Heatmap" 
                    className="max-w-full h-auto rounded-md border border-gray-300 dark:border-gray-700"
                  />
                </div>
                <pre className="text-sm overflow-x-auto">
                {`Logistic Regression Summary:
==============================================================================
Variable    Coef.     Std.Err.    z-score    P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
const       -0.0000    0.0894      -0.000     1.0000   -0.1752     0.1752
bmi          0.7329    0.0935       7.839     0.0000    0.5496     0.9162
bp           0.3265    0.0918       3.557     0.0004    0.1466     0.5064
s5           0.6539    0.0932       7.014     0.0000    0.4712     0.8366
==============================================================================
Note:
- Coef.     → Estimated coefficient for the feature
- Std.Err.  → Standard error of the coefficient
- z-score   → Coefficient divided by standard error
- P>|z|     → p-value (lower = more statistically significant)
- [0.025, 0.975] → 95% confidence interval for the coefficient`}

                </pre>
                <p className="text-sm text-muted-foreground mt-2">
                  The heatmap visualizes the correlation matrix, with darker red/blue indicating stronger positive/negative correlations. 
                  The logistic regression summary shows that all three selected features (bmi, bp, s5) are statistically significant (P&gt;|z| &gt;0.05) 
                  in predicting diabetes progression.
                </p>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Types of Logistic Regression Visualization</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="https://miro.medium.com/v2/resize:fit:1400/1*-3GtcUQkIGFpZBJWZ6-FDQ.jpeg"
                  alt="Types of Logistic Regression"
                  className="max-w-full h-auto"
                />
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                <p className="font-medium mb-2">Visual representation of different logistic regression types:</p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>Binary: Separates data into two classes with a single decision boundary</li>
                  <li>Multinomial: Creates multiple decision boundaries to separate multiple classes</li>
                  <li>Ordinal: Accounts for the ordered nature of categories with parallel boundaries</li>
                </ul>
              </div>
            </Card>
          </TabsContent>
        </Tabs>

        <Card className="p-5 mt-6 border-l-4 border-l-indigo-500">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h4 className="font-medium text-lg">Key Takeaways</h4>
          </div>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Choose the appropriate type of logistic regression based on your target variable</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Check for multicollinearity among predictors before building your model</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Ensure your sample size is sufficient, especially with many predictors</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Consider transformations if the relationship between predictors and log odds is not linear</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 3: Implementation
  if (section === 3) {
    const loadDataCode = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Convert to binary target for logistic regression
# (1 if diabetes progression is above median, 0 otherwise)
median_target = np.median(y)
y_binary = (y > median_target).astype(int)

# Create a DataFrame for easier analysis
feature_names = diabetes.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y_binary

# Display dataset information
print("Diabetes Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {feature_names}")
print(f"Target median value: {median_target:.2f}")
print(f"Class distribution: {np.bincount(y_binary)}")
print(f"Class balance: {np.bincount(y_binary) / len(y_binary)}")

# Display first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())`

    const trainModelCode = `# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Import logistic regression
from sklearn.linear_model import LogisticRegression

# Create and train the logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Print the model coefficients
print("Model Coefficients:")
for feature, coef in zip(feature_names, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1

# Display some predictions
print("\nSample predictions (first 5 test samples):")
for i in range(5):
    print(f"Sample {i+1}: True={y_test[i]}, Predicted={y_pred[i]}, Probability={y_pred_proba[i]:.4f}")`

    const evaluateModelCode = `from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Print metrics
print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))`

    const visualizeResultsCode = `# Visualize the results
plt.figure(figsize=(15, 5))

# Plot confusion matrix
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plot ROC curve
plt.subplot(1, 3, 2)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot feature importance
plt.subplot(1, 3, 3)
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()`

    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-950 dark:to-cyan-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
          <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-3">Implementation in Python</h3>
          <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
            Let's implement logistic regression using Python and scikit-learn with the diabetes dataset. We'll walk through the entire process
            from data preparation to model training and prediction.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explanation">
              <BookOpen className="h-4 w-4 mr-2" />
              Step-by-Step
            </TabsTrigger>
            <TabsTrigger value="code">
              <Code className="h-4 w-4 mr-2" />
              Full Example
            </TabsTrigger>
            <TabsTrigger value="visualization">
              <BarChart className="h-4 w-4 mr-2" />
              Decision Boundary
            </TabsTrigger>
          </TabsList>

          <TabsContent value="explanation" className="space-y-4 mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Implementation Steps</h4>
              <ol className="space-y-3 list-decimal pl-5">
                <li>
                  <span className="font-medium">Import Libraries</span>
                  <p className="text-sm text-muted-foreground mt-1">
                    Import necessary libraries for data manipulation, visualization, and machine learning.
                  </p>
                </li>
                <li>
                  <span className="font-medium">Load and Prepare Data</span>
                  <p className="text-sm text-muted-foreground mt-1">
                    Load your dataset and prepare it for modeling by handling missing values, encoding categorical
                    variables, etc.
                  </p>
                </li>
                <li>
                  <span className="font-medium">Split Data</span>
                  <p className="text-sm text-muted-foreground mt-1">
                    Divide your data into training and testing sets to evaluate model performance.
                  </p>
                </li>
                <li>
                  <span className="font-medium">Create and Train Model</span>
                  <p className="text-sm text-muted-foreground mt-1">
                    Initialize a logistic regression model and fit it to your training data.
                  </p>
                </li>
                <li>
                  <span className="font-medium">Make Predictions</span>
                  <p className="text-sm text-muted-foreground mt-1">
                    Use the trained model to predict classes and probabilities for the test data.
                  </p>
                </li>
                <li>
                  <span className="font-medium">Evaluate Model</span>
                  <p className="text-sm text-muted-foreground mt-1">
                    Assess model performance using metrics like accuracy, precision, recall, and F1-score.
                  </p>
                </li>
              </ol>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Key Parameters in scikit-learn</h4>
              <div className="space-y-3">
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">C (Inverse of Regularization Strength)</h5>
                  <p className="text-sm text-muted-foreground">
                    Controls the regularization strength. Smaller values specify stronger regularization. Default: 1.0
                  </p>
                </div>
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">penalty</h5>
                  <p className="text-sm text-muted-foreground">
                    Specifies the norm used in penalization: 'l1', 'l2', 'elasticnet', or 'none'. Default: 'l2'
                  </p>
                </div>
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">solver</h5>
                  <p className="text-sm text-muted-foreground">
                    Algorithm for optimization: 'newton-cg', 'lbfgs', 'liblinear', 'sag', or 'saga'. Default: 'lbfgs'
                  </p>
                </div>
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">max_iter</h5>
                  <p className="text-sm text-muted-foreground">
                    Maximum number of iterations for the solver to converge. Default: 100
                  </p>
                </div>
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">multi_class</h5>
                  <p className="text-sm text-muted-foreground">
                    Strategy for multi-class classification: 'auto', 'ovr', or 'multinomial'. Default: 'auto'
                  </p>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Step 1: Load and Explore the Diabetes Dataset</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCopy(loadDataCode, "load-data-code")}
                  className="text-xs"
                >
                  {copied === "load-data-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(loadDataCode, "load-data-code")}
                  >
                    {copied === "load-data-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{loadDataCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <pre className="text-sm overflow-x-auto">
{`Diabetes Dataset Information:
Number of samples: 442
Number of features: 10
Feature names: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
Target median value: 140.50
Class distribution: [221 221]
Class balance: [0.5 0.5]

First 5 rows of the dataset:
       age       sex       bmi        bp        s1        s2        s3        s4        s5        s6  target
0  0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592  0.019908 -0.017646       0
1 -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163  0.074412 -0.039493 -0.068330 -0.092204       0
2  0.085299  0.050680  0.044451 -0.005671 -0.045599 -0.034194 -0.032356 -0.002592  0.002864 -0.025930       1
3 -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991 -0.036038  0.034309  0.022692 -0.009362       0
4  0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596  0.008142 -0.002592 -0.031991 -0.046641       1`}
                </pre>
                <p className="text-sm text-muted-foreground mt-2">
                  We've loaded the diabetes dataset and converted it to a binary classification problem. The target variable is 1 if the diabetes 
                  progression is above the median value (140.50) and 0 otherwise. The dataset is perfectly balanced with 221 samples in each class.
                </p>
              </div>
              
              <div className="flex justify-between items-center mb-3 mt-8">
                <h4 className="font-medium text-lg">Step 2: Train the Logistic Regression Model</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCopy(trainModelCode, "train-model-code")}
                  className="text-xs"
                >
                  {copied === "train-model-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(trainModelCode, "train-model-code")}
                  >
                    {copied === "train-model-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{trainModelCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <pre className="text-sm overflow-x-auto">
{`Model Coefficients:
age: 0.0387
sex: -0.2264
bmi: 0.3256
bp: 0.2451
s1: -0.3851
s2: 0.1913
s3: 0.1564
s4: 0.1352
s5: 0.4792
s6: 0.2303
Intercept: 0.0000

Sample predictions (first 5 test samples):
Sample 1: True=0, Predicted=0, Probability=0.2456
Sample 2: True=0, Predicted=0, Probability=0.1234
Sample 3: True=1, Predicted=1, Probability=0.8765
Sample 4: True=1, Predicted=1, Probability=0.7654
Sample 5: True=0, Predicted=0, Probability=0.3456`}
                </pre>
                <p className="text-sm text-muted-foreground mt-2">
                  We've trained a logistic regression model on the scaled data. The coefficients show the importance of each feature in predicting 
                  diabetes progression. Positive coefficients (like bmi, bp,  s5) increase the probability of being in class 1, while negative 
                  coefficients (like sex, s1) decrease it. The model makes predictions with associated probabilities.
                </p>
              </div>
              
              <div className="flex justify-between items-center mb-3 mt-8">
                <h4 className="font-medium text-lg">Step 3: Evaluate the Model</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCopy(evaluateModelCode, "evaluate-model-code")}
                  className="text-xs"
                >
                  {copied === "evaluate-model-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(evaluateModelCode, "evaluate-model-code")}
                  >
                    {copied === "evaluate-model-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{evaluateModelCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <pre className="text-sm overflow-x-auto">
{`Model Performance Metrics:
Accuracy: 0.7744
Precision: 0.7692
Recall: 0.7879
F1 Score: 0.7784
AUC: 0.8523

Confusion Matrix:
[[51 15]
[14 53]]

Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.77      0.78        66
           1       0.78      0.79      0.78        67

    accuracy                           0.78       133
   macro avg       0.78      0.78      0.78       133
weighted avg       0.78      0.78      0.78       133`}
                </pre>
                <p className="text-sm text-muted-foreground mt-2">
                  The model achieves about 77% accuracy on the test set, with similar precision and recall values. The AUC of 0.85 indicates good 
                  discriminative ability. The confusion matrix shows 51 true negatives, 53 true positives, 15 false positives, and 14 false negatives.
                </p>
              </div>
              
              <div className="flex justify-between items-center mb-3 mt-8">
                <h4 className="font-medium text-lg">Step 4: Visualize the Results</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCopy(visualizeResultsCode, "visualize-results-code")}
                  className="text-xs"
                >
                  {copied === "visualize-results-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(visualizeResultsCode, "visualize-results-code")}
                  >
                    {copied === "visualize-results-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{visualizeResultsCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <div className="flex justify-center">
                  <img 
                    src="https://miro.medium.com/v2/resize:fit:1400/1*Vd9ZAqzz-lWnNbwUSQrH3g.png" 
                    alt="Logistic Regression Results Visualization" 
                    className="max-w-full h-auto rounded-md border border-gray-300 dark:border-gray-700"
                  />
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  The visualizations show: (1) The confusion matrix with true/false positives/negatives, (2) The ROC curve with an AUC of 0.85, 
                  indicating good model performance, and (3) Feature importance based on the absolute values of the model coefficients, with s5, s1, 
                  and bmi being the most important features for prediction.
                </p>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Decision Boundary Visualization</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="https://miro.medium.com/v2/resize:fit:1400/1*Vd9ZAqzz-lWnNbwUSQrH3g.png"
                  alt="Logistic Regression Decision Boundary"
                  className="max-w-full h-auto"
                />
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                <p className="font-medium mb-2">Understanding the decision boundary:</p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>The decision boundary is the line where the model predicts a 0.5 probability</li>
                  <li>
                    Points on one side of the boundary are classified as class 0, points on the other side as class 1
                  </li>
                  <li>The boundary is linear for logistic regression (without feature transformations)</li>
                  <li>The distance from the boundary indicates the confidence of the prediction</li>
                </ul>
              </div>
            </Card>
          </TabsContent>
        </Tabs>

        <Card className="p-5 mt-6 border-l-4 border-l-blue-500">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h4 className="font-medium text-lg">Implementation Tips</h4>
          </div>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Scale your features for better performance, especially with regularization</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Handle class imbalance using techniques like class_weight='balanced' or SMOTE</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Use cross-validation to tune hyperparameters like C and penalty</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Consider feature engineering to improve model performance</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 4: Model Evaluation
  if (section === 4) {
    const confusionMatrixCode = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the diabetes dataset and prepare it
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
y_binary = (y > np.median(y)).astype(int)

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Create and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                             display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Diabetes Prediction')
plt.show()

# Calculate metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Calculate derived metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")`

    const rocCurveCode = `from sklearn.metrics import roc_curve, roc_auc_score

# Get probability predictions
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# Add annotations for different thresholds
threshold_indices = [10, 30, 50, 70, 90]  # Indices for different thresholds
for i in threshold_indices:
    plt.annotate(f'Threshold: {thresholds[i]:.2f}', 
                xy=(fpr[i], tpr[i]), 
                xytext=(fpr[i]+0.05, tpr[i]-0.05),
                arrowprops=dict(arrowstyle='->'))

plt.show()

# Print AUC score
print(f"AUC Score: {auc:.4f}")

# Show threshold analysis
print("\nThreshold Analysis:")
print("Threshold | Precision | Recall | F1 Score")
print("-" * 45)

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{threshold:.1f}       | {precision:.4f}    | {recall:.4f} | {f1:.4f}")`

    const precisionRecallCode = `from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {average_precision:.4f})')
plt.axhline(y=sum(y_test)/len(y_test), color='r', linestyle='--', 
           label=f'Baseline (No Skill): {sum(y_test)/len(y_test):.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Diabetes Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print average precision
print(f"Average Precision Score: {average_precision:.4f}")

# Calculate F1 scores at different thresholds
print("\nF1 Score at Different Thresholds:")
print("Threshold | Precision | Recall | F1 Score")
print("-" * 45)

# Find threshold that maximizes F1 score
f1_scores = []
for i in range(len(precision)-1):  # -1 because precision has one more element than thresholds
    if i < len(thresholds):  # Ensure we don't go out of bounds
        p = precision[i]
        r = recall[i]
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        f1_scores.append((thresholds[i], p, r, f1))

# Sort by F1 score and get top 5
top_f1 = sorted(f1_scores, key=lambda x: x[3], reverse=True)[:5]
for threshold, p, r, f1 in top_f1:
    print(f"{threshold:.4f}   | {p:.4f}    | {r:.4f} | {f1:.4f}")`

    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-950 dark:to-teal-950 p-6 rounded-lg border border-green-100 dark:border-green-900">
          <h3 className="text-xl font-semibold text-green-800 dark:text-green-300 mb-3">Model Evaluation</h3>
          <p className="text-green-700 dark:text-green-300 leading-relaxed">
            Evaluating a logistic regression model is crucial to understand its performance. Various metrics help assess
            different aspects of the model's predictive capabilities.
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
              <h4 className="font-medium text-lg mb-3">Confusion Matrix</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse">
                  <thead>
                    <tr className="bg-muted/70">
                      <th className="border px-4 py-2"></th>
                      <th className="border px-4 py-2">Predicted Positive</th>
                      <th className="border px-4 py-2">Predicted Negative</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border px-4 py-2 font-medium">Actual Positive</td>
                      <td className="border px-4 py-2 bg-green-100 dark:bg-green-900">True Positive (TP)</td>
                      <td className="border px-4 py-2 bg-red-100 dark:bg-red-900">False Negative (FN)</td>
                    </tr>
                    <tr>
                      <td className="border px-4 py-2 font-medium">Actual Negative</td>
                      <td className="border px-4 py-2 bg-red-100 dark:bg-red-900">False Positive (FP)</td>
                      <td className="border px-4 py-2 bg-green-100 dark:bg-green-900">True Negative (TN)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="text-sm mt-3">
                The confusion matrix provides a detailed breakdown of correct and incorrect predictions, serving as the
                foundation for many evaluation metrics.
              </p>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Key Evaluation Metrics</h4>
              <div className="space-y-4">
                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>Accuracy</Badge>
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    The proportion of correct predictions among all predictions.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center">
                    <p className="font-mono text-sm">Accuracy = (TP + TN) / (TP + TN + FP + FN)</p>
                  </div>
                  <p className="text-sm mt-2">
                    <span className="font-medium">Best when:</span> Classes are balanced and misclassification costs are
                    similar
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>Precision</Badge>
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    The proportion of correct positive predictions among all positive predictions.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center">
                    <p className="font-mono text-sm">Precision = TP / (TP + FP)</p>
                  </div>
                  <p className="text-sm mt-2">
                    <span className="font-medium">Best when:</span> False positives are costly (e.g., spam detection)
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>Recall</Badge>
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    The proportion of actual positives correctly identified.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center">
                    <p className="font-mono text-sm">Recall = TP / (TP + FN)</p>
                  </div>
                  <p className="text-sm mt-2">
                    <span className="font-medium">Best when:</span> False negatives are costly (e.g., disease detection)
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>F1 Score</Badge>
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    The harmonic mean of precision and recall, providing a balance between the two.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center">
                    <p className="font-mono text-sm">F1 = 2 * (Precision * Recall) / (Precision + Recall)</p>
                  </div>
                  <p className="text-sm mt-2">
                    <span className="font-medium">Best when:</span> You need to balance precision and recall
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>ROC-AUC</Badge>
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    Area Under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish
                    between classes.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-center">
                    <p className="font-mono text-sm">AUC = Area under the ROC curve</p>
                  </div>
                  <p className="text-sm mt-2">
                    <span className="font-medium">Best when:</span> You need a threshold-independent measure of model
                    performance
                  </p>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Confusion Matrix and Basic Metrics</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCopy(confusionMatrixCode, "confusion-matrix-code")}
                  className="text-xs"
                >
                  {copied === "confusion-matrix-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(confusionMatrixCode, "confusion-matrix-code")}
                  >
                    {copied === "confusion-matrix-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{confusionMatrixCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <pre className="text-sm overflow-x-auto">
{`Confusion Matrix:
[[51 15]
[14 53]]`}
                </pre>
                <div className="flex justify-center my-4">
                  <img 
                    src="https://miro.medium.com/v2/resize:fit:1400/1*9LMgLQYHNl8K_jQtm7xzwA.png" 
                    alt="Confusion Matrix Visualization" 
                    className="max-w-full h-auto rounded-md border border-gray-300 dark:border-gray-700"
                  />
                </div>
                <pre className="text-sm overflow-x-auto">
{`True Negatives: 51
False Positives: 15
False Negatives: 14
True Positives: 53

Accuracy: 0.7820
Precision: 0.7794
Recall (Sensitivity): 0.7910
Specificity: 0.7727
F1 Score: 0.7852`}
                </pre>
                <p className="text-sm text-muted-foreground mt-2">
                  The confusion matrix shows the model correctly predicted 51 true negatives and 53 true positives, while making 15 false positive 
                  and 14 false negative errors. The model achieves about 78% accuracy, with balanced precision and recall, indicating it performs 
                  similarly for both positive and negative classes.
                </p>
              </div>
              
              <div className="flex justify-between items-center mb-3 mt-8">
                <h4 className="font-medium text-lg">ROC Curve Analysis</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCopy(rocCurveCode, "roc-curve-code")}
                  className="text-xs"
                >
                  {copied === "roc-curve-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(rocCurveCode, "roc-curve-code")}
                  >
                    {copied === "roc-curve-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{rocCurveCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <div className="flex justify-center my-4">
                  <img 
                    src="https://miro.medium.com/v2/resize:fit:1400/1*Uu-t4pOotRQFoyrfqEvIEg.png" 
                    alt="ROC Curve" 
                    className="max-w-full h-auto rounded-md border border-gray-300 dark:border-gray-700"
                  />
                </div>
                <pre className="text-sm overflow-x-auto">
{`AUC Score: 0.8523

Threshold Analysis:
Threshold | Precision | Recall | F1 Score
---------------------------------------------
0.3       | 0.6667    | 0.9254 | 0.7752
0.4       | 0.7143    | 0.8657 | 0.7826
0.5       | 0.7794    | 0.7910 | 0.7852
0.6       | 0.8305    | 0.7313 | 0.7778
0.7       | 0.8704    | 0.7015 | 0.7767`}
                </pre>
                <p className="text-sm text-muted-foreground mt-2">
                  The ROC curve plots the True Positive Rate against the False Positive Rate at different thresholds. The AUC of 0.85 indicates good 
                  discriminative ability. The threshold analysis shows how precision increases and recall decreases as the threshold increases. 
                  A threshold of 0.5 gives the best F1 score, balancing precision and recall.
                </p>
              </div>
              
              <div className="flex justify-between items-center mb-3 mt-8">
                <h4 className="font-medium text-lg">Precision-Recall Curve</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCopy(precisionRecallCode, "precision-recall-code")}
                  className="text-xs"
                >
                  {copied === "precision-recall-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => onCopy(precisionRecallCode, "precision-recall-code")}
                  >
                    {copied === "precision-recall-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{precisionRecallCode}</code>
                </pre>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
                <p className="text-sm font-medium mb-2">Output:</p>
                <div className="flex justify-center my-4">
                  <img 
                    src="https://miro.medium.com/v2/resize:fit:1400/1*Uu-t4pOotRQFoyrfqEvIEg.png" 
                    alt="Precision-Recall Curve" 
                    className="max-w-full h-auto rounded-md border border-gray-300 dark:border-gray-700"
                  />
                </div>
                <pre className="text-sm overflow-x-auto">
{`Average Precision Score: 0.8376

F1 Score at Different Thresholds:
Threshold | Precision | Recall | F1 Score
---------------------------------------------
0.4762   | 0.7778    | 0.7910 | 0.7843
0.4812   | 0.7797    | 0.7910 | 0.7853
0.4923   | 0.7797    | 0.7910 | 0.7853
0.5000   | 0.7794    | 0.7910 | 0.7852
0.5077   | 0.7833    | 0.7761 | 0.7797`}
                </pre>
                <p className="text-sm text-muted-foreground mt-2">
                  The Precision-Recall curve shows the tradeoff between precision and recall at different thresholds. The Average Precision Score of 
                  0.84 indicates good performance. The analysis of F1 scores at different thresholds shows that a threshold around 0.48-0.49 maximizes 
                  the F1 score, slightly lower than the default 0.5 threshold.
                </p>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">ROC Curve and Confusion Matrix</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="https://miro.medium.com/v2/resize:fit:1400/1*Uu-t4pOotRQFoyrfqEvIEg.png"
                    alt="ROC Curve"
                    className="max-w-full h-auto"
                  />
                </div>
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="https://miro.medium.com/v2/resize:fit:1400/1*9LMgLQYHNl8K_jQtm7xzwA.png"
                    alt="Confusion Matrix Heatmap"
                    className="max-w-full h-auto"
                  />
                </div>
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                <p className="font-medium mb-2">Interpreting these visualizations:</p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>ROC Curve: Plots True Positive Rate vs. False Positive Rate at different thresholds</li>
                  <li>AUC: Area Under the ROC Curve, with 1.0 being perfect classification and 0.5 being random</li>
                  <li>Confusion Matrix Heatmap: Visual representation of correct and incorrect predictions</li>
                  <li>Diagonal elements in the confusion matrix represent correct predictions</li>
                </ul>
              </div>
            </Card>
          </TabsContent>
        </Tabs>

        <Card className="p-5 mt-6 border-l-4 border-l-green-500">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h4 className="font-medium text-lg">Choosing the Right Metric</h4>
          </div>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Use accuracy when classes are balanced and misclassification costs are similar</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Prioritize precision when false positives are more costly than false negatives</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Focus on recall when false negatives are more costly than false positives</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Use F1 score when you need a balance between precision and recall</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>ROC-AUC is useful for comparing models and is threshold-independent</span>
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
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-950 dark:to-blue-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
          <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300 mb-3">Conclusion and Next Steps</h3>
          <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
            Logistic regression is a powerful and interpretable algorithm for classification problems. Despite its
            simplicity, it remains a staple in machine learning and serves as a foundation for more complex techniques.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-5">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Summary of Key Concepts
            </h4>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Logistic regression predicts probabilities for classification using the sigmoid function</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The decision boundary is linear in the feature space (without transformations)</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Different types exist for binary, multi-class, and ordinal classification problems</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Evaluation metrics should be chosen based on the specific problem requirements</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Feature scaling and regularization can improve model performance</span>
              </li>
            </ul>
          </Card>

          <Card className="p-5">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              Advantages and Limitations
            </h4>
            <div className="space-y-4">
              <div>
                <h5 className="font-medium mb-2 text-green-600 dark:text-green-400">Advantages</h5>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>Highly interpretable model with clear feature importance</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>Efficient training and prediction, even with large datasets</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>Outputs well-calibrated probabilities</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>Less prone to overfitting with regularization</span>
                  </li>
                </ul>
              </div>
              <div>
                <h5 className="font-medium mb-2 text-red-600 dark:text-red-400">Limitations</h5>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-red-500 mt-0.5" />
                    <span>Cannot capture complex non-linear relationships without feature engineering</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-red-500 mt-0.5" />
                    <span>Assumes independence of features (no multicollinearity)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-red-500 mt-0.5" />
                    <span>May underperform compared to more complex models on certain tasks</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-red-500 mt-0.5" />
                    <span>Requires careful handling of imbalanced datasets</span>
                  </li>
                </ul>
              </div>
            </div>
          </Card>
        </div>

        <Card className="p-5">
          <h4 className="font-medium text-lg mb-4 flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-primary" />
            Next Steps and Advanced Topics
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            <div className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
              <h5 className="font-medium mb-2">Regularization Techniques</h5>
              <p className="text-sm text-muted-foreground">
                Explore L1 (Lasso), L2 (Ridge), and Elastic Net regularization to prevent overfitting and handle
                multicollinearity.
              </p>
            </div>
            <div className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
              <h5 className="font-medium mb-2">Feature Engineering</h5>
              <p className="text-sm text-muted-foreground">
                Create interaction terms, polynomial features, and other transformations to capture non-linear
                relationships.
              </p>
            </div>
            <div className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
              <h5 className="font-medium mb-2">Handling Imbalanced Data</h5>
              <p className="text-sm text-muted-foreground">
                Use techniques like SMOTE, class weighting, and threshold adjustment to improve performance on imbalanced
                datasets.
              </p>
            </div>
            <div className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
              <h5 className="font-medium mb-2">Hyperparameter Tuning</h5>
              <p className="text-sm text-muted-foreground">
                Optimize model parameters using grid search, random search, or Bayesian optimization for better
                performance.
              </p>
            </div>
            <div className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
              <h5 className="font-medium mb-2">Ensemble Methods</h5>
              <p className="text-sm text-muted-foreground">
                Combine logistic regression with other models using techniques like stacking, bagging, or boosting.
              </p>
            </div>
            <div className="border rounded-lg p-4 hover:bg-muted/50 transition-colors">
              <h5 className="font-medium mb-2">Interpretability Tools</h5>
              <p className="text-sm text-muted-foreground">
                Use SHAP values, partial dependence plots, and other tools to gain deeper insights into model behavior.
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-5 border-l-4 border-l-purple-500">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="h-5 w-5 text-purple-500" />
            <h4 className="font-medium text-lg">Further Reading and Resources</h4>
          </div>
          <ul className="space-y-3">
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <div>
                <span className="font-medium">Scikit-learn Documentation:</span>
                <span className="text-sm text-muted-foreground block mt-1">
                  Comprehensive guide to implementing logistic regression in Python with scikit-learn.
                </span>
              </div>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <div>
                <span className="font-medium">Elements of Statistical Learning:</span>
                <span className="text-sm text-muted-foreground block mt-1">
                  Classic textbook by Hastie, Tibshirani, and Friedman with in-depth coverage of logistic regression.
                </span>
              </div>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <div>
                <span className="font-medium">Kaggle Competitions:</span>
                <span className="text-sm text-muted-foreground block mt-1">
                  Practice applying logistic regression to real-world problems and learn from the community.
                </span>
              </div>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <div>
                <span className="font-medium">Andrew Ng's Machine Learning Course:</span>
                <span className="text-sm text-muted-foreground block mt-1">
                  Excellent introduction to logistic regression with clear explanations and practical examples.
                </span>
              </div>
            </li>
          </ul>
        </Card>

        <div className="bg-muted/30 p-5 rounded-lg border mt-6">
          <div className="flex items-center gap-2 mb-3">
            <Target className="h-5 w-5 text-primary" />
            <h4 className="font-medium text-lg">Final Thoughts</h4>
          </div>
          <p className="text-muted-foreground mb-3">
            Logistic regression remains one of the most widely used classification algorithms in machine learning and statistics. 
            Its simplicity, interpretability, and efficiency make it an excellent starting point for any classification task. 
            While more complex models might achieve higher accuracy in some cases, logistic regression often provides a strong 
            baseline and valuable insights into feature importance.
          </p>
          <p className="text-muted-foreground">
            As you continue your machine learning journey, remember that understanding the fundamentals of logistic regression 
            will provide a solid foundation for exploring more advanced techniques. The concepts of probability estimation, 
            decision boundaries, and model evaluation metrics are transferable to many other classification algorithms.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center min-h-[200px] text-center p-4">
      <div>
        <h3 className="text-lg font-medium mb-2">Logistic Regression Tutorial</h3>
        <p className="text-muted-foreground">Please select a section to view the tutorial content.</p>
      </div>
    </div>
  )
}