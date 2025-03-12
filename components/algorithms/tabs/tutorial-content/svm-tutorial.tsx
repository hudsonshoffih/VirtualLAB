"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { BookOpen, Code, BarChart, Lightbulb, CheckCircle, ArrowRight, ChevronLeft, ChevronRight, Copy, Check, LineChart, GitMerge, Maximize2, Minimize2, Zap, Settings, Sigma, Workflow } from 'lucide-react'
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
    ChartTooltipItem,
    ChartBar,
    ChartBarItem,
    ChartTitle,
  } from "@/components/ui/chart"

  interface svmTutorialProps {
    section: number
    onCopy: (text: string, id: string) => void
    copied: string | null
  }
  
  export function SvmTutorial({ section, onCopy, copied }: svmTutorialProps) {
    const [activeTab, setActiveTab] = useState("explanation")

  // Render content based on current section
  const [copiedState, setCopiedState] = useState<string | null>(null);

const featureImportanceSVMCode = `// Add your feature importance SVM code here`;
const randomSearchSVMCode = `# Example code for Randomized Search CV with SVM`;
const svmClassificationCode = `// Add your SVM classification code here`;
const decisionBoundaryCode = `// Add your decision boundary code here`;
const svmRegressionCode = `// Add your SVM regression code here`;

const gridSearchSVMCode = `# Example code for Grid Search CV with SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Create a base model
svc = SVC()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
`;
    // Section 0: Introduction to SVM
    if (section === 0) {
      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-950 dark:to-purple-950 p-6 rounded-lg border border-violet-100 dark:border-violet-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-violet-100 dark:bg-violet-900 p-2 rounded-full">
                <GitMerge className="h-6 w-6 text-violet-600 dark:text-violet-400" />
              </div>
              <h3 className="text-xl font-semibold text-violet-800 dark:text-violet-300">
                What is Support Vector Machine?
              </h3>
            </div>
            <p className="text-violet-700 dark:text-violet-300 leading-relaxed">
              Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and
              regression tasks. It works by finding the optimal hyperplane that maximally separates different classes in
              the feature space. SVM is particularly effective for complex, high-dimensional datasets and can handle
              both linearly and non-linearly separable data through the use of kernel functions.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="h-5 w-5 text-violet-500" />
                <h4 className="font-medium text-lg">Key Concepts</h4>
              </div>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="bg-violet-100 dark:bg-violet-900 text-violet-600 dark:text-violet-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    1
                  </span>
                  <span>Hyperplane (decision boundary)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-violet-100 dark:bg-violet-900 text-violet-600 dark:text-violet-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    2
                  </span>
                  <span>Margin (distance between hyperplane and closest data points)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-violet-100 dark:bg-violet-900 text-violet-600 dark:text-violet-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    3
                  </span>
                  <span>Support vectors (data points that define the margin)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-violet-100 dark:bg-violet-900 text-violet-600 dark:text-violet-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    4
                  </span>
                  <span>Kernel trick (for non-linear classification)</span>
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
                  <span>Text and document classification</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Image recognition and computer vision</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Bioinformatics and genomic data analysis</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>Financial forecasting and risk assessment</span>
                </li>
              </ul>
            </Card>
          </div>

          <div className="mt-6">
            <h4 className="font-medium text-lg mb-3 flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              What You'll Learn
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {[
                { icon: <Sigma className="h-4 w-4" />, text: "Mathematical Foundations" },
                { icon: <LineChart className="h-4 w-4" />, text: "Linear SVM" },
                { icon: <Workflow className="h-4 w-4" />, text: "Kernel Trick" },
                { icon: <Code className="h-4 w-4" />, text: "Implementation with Scikit-Learn" },
                { icon: <Settings className="h-4 w-4" />, text: "Hyperparameter Tuning" },
                { icon: <BarChart className="h-4 w-4" />, text: "Model Evaluation" },
                { icon: <GitMerge className="h-4 w-4" />, text: "Multi-class Classification" },
                { icon: <Zap className="h-4 w-4" />, text: "Practical Applications" },
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
                <span>Basic understanding of Python programming</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-1 text-primary" />
                <span>Familiarity with fundamental machine learning concepts</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-1 text-primary" />
                <span>Basic knowledge of linear algebra and calculus</span>
              </li>
            </ul>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-950 dark:to-purple-950 p-6 rounded-lg border border-violet-100 dark:border-violet-900">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-violet-100 dark:bg-violet-900 p-2 rounded-full">
                  <Maximize2 className="h-5 w-5 text-violet-600 dark:text-violet-400" />
                </div>
                <h3 className="text-lg font-semibold text-violet-800 dark:text-violet-300">
                  Advantages of SVM
                </h3>
              </div>
              <ul className="space-y-3 text-violet-700 dark:text-violet-300">
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Effective in high-dimensional spaces</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Memory efficient (uses subset of training points)</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Versatile through different kernel functions</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Robust against overfitting in high-dimensional spaces</span>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                  <span>Theoretically well-founded mathematical framework</span>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950 dark:to-orange-950 p-6 rounded-lg border border-amber-100 dark:border-amber-900">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-amber-100 dark:bg-amber-900 p-2 rounded-full">
                  <Minimize2 className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                </div>
                <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-300">Limitations to Consider</h3>
              </div>
              <ul className="space-y-3 text-amber-700 dark:text-amber-300">
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Computationally intensive for large datasets</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Choosing the right kernel and parameters can be challenging</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Does not directly provide probability estimates</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Less effective when classes overlap significantly</span>
                </li>
                <li className="flex items-start gap-2">
                  <ArrowRight className="h-4 w-4 text-amber-500 mt-1" />
                  <span>Sensitive to noise and outliers in training data</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-muted/20 p-6 rounded-lg border mt-6">
            <h4 className="font-medium text-lg mb-4 flex items-center gap-2">
              <GitMerge className="h-5 w-5 text-primary" />
              SVM at a Glance
            </h4>
            <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
              <img
                src="/placeholder.svg?height=400&width=800"
                alt="SVM Overview"
                className="max-w-full h-auto"
                onError={(e) => {
                  e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*TudH6YvvH7-h5ZyF2dJV2w.jpeg"
                }}
              />
            </div>
            <p className="mt-4 text-sm text-muted-foreground">
              The diagram illustrates how SVM works: it finds the optimal hyperplane (decision boundary) that maximizes
              the margin between different classes. The data points that lie on the margin boundaries are called support
              vectors, and they are the only ones that influence the position and orientation of the hyperplane.
            </p>
          </div>
        </div>
      )
    }

    // Section 1: Mathematical Foundations
    if (section === 1) {
      function copyToClipboard(svmMathCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-950 dark:to-purple-950 p-6 rounded-lg border border-violet-100 dark:border-violet-900">
            <h3 className="text-xl font-semibold text-violet-800 dark:text-violet-300 mb-3">
              Mathematical Foundations
            </h3>
            <p className="text-violet-700 dark:text-violet-300 leading-relaxed">
              Understanding the mathematical principles behind Support Vector Machines is essential to grasp how they
              work. SVM is based on the concept of finding the optimal hyperplane that maximizes the margin between
              different classes, which involves principles from optimization theory and linear algebra.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Core Concepts
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Mathematical Formulation
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Visualization
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Key Mathematical Concepts</h4>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="bg-violet-100 dark:bg-violet-900 text-violet-600 dark:text-violet-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Hyperplane</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        In a d-dimensional space, a hyperplane is a flat subspace of dimension d-1. For a 2D space, it's
                        a line; for a 3D space, it's a plane. Mathematically, it's defined as:
                        <span className="block mt-2 text-center font-mono">w·x + b = 0</span>
                        where w is the normal vector to the hyperplane, x is a point on the hyperplane, and b is the
                        bias term.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-violet-100 dark:bg-violet-900 text-violet-600 dark:text-violet-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Margin</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        The margin is the distance between the hyperplane and the closest data points from each class.
                        SVM aims to maximize this margin, which is calculated as 2/||w||, where ||w|| is the magnitude
                        of the normal vector.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-violet-100 dark:bg-violet-900 text-violet-600 dark:text-violet-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Support Vectors</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Support vectors are the data points that lie closest to the decision boundary. They are the
                        critical elements in defining the hyperplane and margin. Only these points affect the position
                        of the hyperplane, making SVM memory efficient.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-violet-100 dark:bg-violet-900 text-violet-600 dark:text-violet-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Functional and Geometric Margins</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        The functional margin is defined as y(w·x + b), where y is the class label (+1 or -1). The
                        geometric margin is the actual distance from a point to the hyperplane, calculated as the
                        functional margin divided by ||w||.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Optimization Problem</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  SVM formulates the task of finding the optimal hyperplane as a constrained optimization problem:
                </p>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <p className="font-mono text-center">
                    Minimize: ||w||²/2
                    <br />
                    Subject to: y<sub>i</sub>(w·x<sub>i</sub> + b) ≥ 1 for all i
                  </p>
                </div>
                <p className="text-sm text-muted-foreground mt-4">
                  This optimization problem aims to minimize the norm of w (which maximizes the margin) while ensuring
                  that all data points are correctly classified with a margin of at least 1/||w||.
                </p>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Lagrangian Formulation</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  The constrained optimization problem is typically solved using Lagrange multipliers, transforming it
                  into the following dual form:
                </p>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <p className="font-mono text-center">
                    Maximize: Σα<sub>i</sub> - (1/2)ΣΣα<sub>i</sub>α<sub>j</sub>y<sub>i</sub>y<sub>j</sub>x<sub>i</sub>·x
                    <sub>j</sub>
                    <br />
                    Subject to: α<sub>i</sub> ≥ 0 and Σα<sub>i</sub>y<sub>i</sub> = 0
                  </p>
                </div>
                <p className="text-sm text-muted-foreground mt-4">
                  In this dual form, α<sub>i</sub> are the Lagrange multipliers. Only the support vectors have non-zero
                  α<sub>i</sub> values, which is why SVM depends only on these points for making predictions.
                </p>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Soft Margin SVM</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  In practice, data may not be perfectly separable. Soft margin SVM introduces slack variables (ξ<sub>i</sub>)
                  to allow for some misclassification:
                </p>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <p className="font-mono text-center">
                    Minimize: ||w||²/2 + C·Σξ<sub>i</sub>
                    <br />
                    Subject to: y<sub>i</sub>(w·x<sub>i</sub> + b) ≥ 1 - ξ<sub>i</sub> and ξ<sub>i</sub> ≥ 0 for all i
                  </p>
                </div>
                <p className="text-sm text-muted-foreground mt-4">
                  The parameter C controls the trade-off between maximizing the margin and minimizing the
                  classification error. A larger C places more emphasis on correctly classifying all training examples,
                  potentially leading to a smaller margin and overfitting.
                </p>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Mathematical Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(svmMathCode, "svm-math-code")}
                    className="text-xs"
                  >
                    {copiedState === "svm-math-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{svmMathCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates the mathematical principles of SVM by implementing a simplified version of
                    the algorithm using quadratic programming to solve the optimization problem. It shows how to find
                    the optimal hyperplane, calculate the margin, and identify support vectors.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Hyperplane and Margin Visualization</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="SVM Hyperplane and Margin"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*ZT9FaJ_gve0cypjaDJ4Y_w.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The visualization shows:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>The optimal hyperplane (solid line) separating two classes</li>
                    <li>The margin boundaries (dashed lines) on either side of the hyperplane</li>
                    <li>Support vectors (highlighted points) that define the margin</li>
                    <li>The geometric interpretation of the margin as the distance between the hyperplane and support vectors</li>
                  </ul>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Hard vs. Soft Margin</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-medium mb-2 text-center">Hard Margin SVM</h5>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=300&width=300"
                        alt="Hard Margin SVM"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:600/1*Fjj7EblDs2J88GgJmyKL8w.png"
                        }}
                      />
                    </div>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Hard margin SVM requires perfect separation of classes with no misclassification. It works well for linearly separable data but is sensitive to outliers.
                    </p>
                  </div>
                  <div>
                    <h5 className="font-medium mb-2 text-center">Soft Margin SVM</h5>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=300&width=300"
                        alt="Soft Margin SVM"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*NrSJ8uYwLTUl2iIWNzEEOQ.png"
                        }}
                      />
                    </div>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Soft margin SVM allows some misclassification by introducing slack variables, making it more robust to noise and outliers in the data.
                    </p>
                  </div>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-violet-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>SVM finds the optimal hyperplane that maximizes the margin between classes</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Support vectors are the data points closest to the hyperplane that define the margin</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The optimization problem can be solved using Lagrange multipliers</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Soft margin SVM introduces slack variables to handle non-separable data</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The C parameter controls the trade-off between margin size and classification error</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 2: Linear SVM
    if (section === 2) {
      function copyToClipboard(linearSVMCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
            <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-3">Linear SVM</h3>
            <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
              Linear SVM is the simplest form of Support Vector Machine, used when the data is linearly separable or
              nearly linearly separable. It finds a straight hyperplane (a line in 2D, a plane in 3D) to separate the
              classes. Let's explore how linear SVM works and when to use it.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                How It Works
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
                <h4 className="font-medium text-lg mb-3">Linear SVM Explained</h4>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Linear Decision Boundary</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Linear SVM creates a straight-line decision boundary (or hyperplane in higher dimensions) to
                        separate data points of different classes. The equation of this hyperplane is:
                        <span className="block mt-2 text-center font-mono">w·x + b = 0</span>
                        where w is the weight vector, x is the input vector, and b is the bias term.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Classification Rule</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        For a new data point x, the classification is determined by:
                        <span className="block mt-2 text-center font-mono">
                          f(x) = sign(w·x + b)
                          <br />
                          If f(x) &gt; 0, classify as positive class
                          <br />
                          If f(x) &lt; 0, classify as negative class
                        </span>
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Margin Maximization</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Linear SVM finds the hyperplane that maximizes the margin between the two classes. The margin is
                        the distance between the hyperplane and the closest data points from each class (the support
                        vectors).
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Hard vs. Soft Margin</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Hard margin SVM requires perfect separation of classes, while soft margin SVM allows some
                        misclassification by introducing slack variables and a penalty parameter C. Soft margin is more
                        practical for real-world data that may not be perfectly separable.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">When to Use Linear SVM</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4 bg-green-50 dark:bg-green-950">
                    <h5 className="font-medium mb-2">Good For</h5>
                    <ul className="text-sm space-y-2 list-disc pl-5">
                      <li>Linearly separable data</li>
                      <li>High-dimensional spaces (e.g., text classification)</li>
                      <li>When the number of features is greater than the number of samples</li>
                      <li>When you need a simple, interpretable model</li>
                      <li>Memory-constrained environments (only stores support vectors)</li>
                    </ul>
                  </div>
                  <div className="border rounded-lg p-4 bg-red-50 dark:bg-red-950">
                    <h5 className="font-medium mb-2">Less Effective For</h5>
                    <ul className="text-sm space-y-2 list-disc pl-5">
                      <li>Non-linearly separable data</li>
                      <li>Very large datasets (computationally intensive)</li>
                      <li>When probabilistic outputs are required</li>
                      <li>Highly imbalanced datasets</li>
                      <li>When feature scaling hasn't been applied</li>
                    </ul>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">The C Parameter</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  The C parameter in soft margin SVM controls the trade-off between achieving a wide margin and
                  minimizing classification error:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2">Small C Value</h5>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Prioritizes a wider margin</li>
                      <li>Allows more misclassifications</li>
                      <li>Less sensitive to individual data points</li>
                      <li>Helps prevent overfitting</li>
                      <li>Good for noisy data</li>
                    </ul>
                  </div>
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2">Large C Value</h5>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Prioritizes correct classification</li>
                      <li>Creates a narrower margin</li>
                      <li>More sensitive to individual data points</li>
                      <li>May lead to overfitting</li>
                      <li>Good for clean, well-separated data</li>
                    </ul>
                  </div>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Linear SVM Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(linearSVMCode, "linear-svm-code")}
                    className="text-xs"
                  >
                    {copiedState === "linear-svm-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{linearSVMCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how to implement a linear SVM using scikit-learn. It includes data loading,
                    preprocessing, model training, and evaluation. The example also shows how to visualize the decision
                    boundary and support vectors.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Linear SVM Decision Boundary</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Linear SVM Decision Boundary"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_margin_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The visualization shows:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>The linear decision boundary (solid line) separating two classes</li>
                    <li>The margin boundaries (dashed lines) on either side of the decision boundary</li>
                    <li>Support vectors (circled points) that define the margin</li>
                    <li>Data points from different classes shown in different colors</li>
                  </ul>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Effect of C Parameter</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Effect of C Parameter"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_regularization_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">This visualization demonstrates how the C parameter affects the decision boundary:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Small C values (left) create a wider margin but allow more misclassifications</li>
                    <li>Large C values (right) create a narrower margin with fewer misclassifications</li>
                    <li>The optimal C value depends on the specific dataset and the desired trade-off between margin width and classification accuracy</li>
                  </ul>
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
                <span>Linear SVM creates a straight-line decision boundary to separate classes</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>It works best for linearly separable data or high-dimensional spaces</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The C parameter controls the trade-off between margin width and classification error</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Support vectors are the only data points that influence the decision boundary</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>For non-linearly separable data, kernel methods (next section) are more appropriate</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 3: Kernel Trick
    if (section === 3) {
      function copyToClipboard(kernelSVMCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-teal-50 to-emerald-50 dark:from-teal-950 dark:to-emerald-950 p-6 rounded-lg border border-teal-100 dark:border-teal-900">
            <h3 className="text-xl font-semibold text-teal-800 dark:text-teal-300 mb-3">Kernel Trick</h3>
            <p className="text-teal-700 dark:text-teal-300 leading-relaxed">
              The kernel trick is a powerful technique that allows SVMs to handle non-linearly separable data. It
              implicitly maps the input data into a higher-dimensional space where it becomes linearly separable,
              without actually computing the coordinates in that space. This makes SVMs versatile for complex
              classification tasks.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                How It Works
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
                <h4 className="font-medium text-lg mb-3">The Kernel Trick Explained</h4>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="bg-teal-100 dark:bg-teal-900 text-teal-600 dark:text-teal-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Feature Space Transformation</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        The kernel trick implicitly maps data from the original input space to a higher-dimensional
                        feature space where linear separation becomes possible. This is done without explicitly
                        calculating the coordinates in the higher-dimensional space.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-teal-100 dark:bg-teal-900 text-teal-600 dark:text-teal-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Kernel Functions</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        A kernel function K(x, y) computes the dot product of two vectors in the transformed feature
                        space without explicitly computing the transformation:
                        <span className="block mt-2 text-center font-mono">K(x, y) = φ(x)·φ(y)</span>
                        where φ is the mapping function to the higher-dimensional space.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-teal-100 dark:bg-teal-900 text-teal-600 dark:text-teal-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Computational Efficiency</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        The kernel trick makes SVM computationally efficient for non-linear classification. Instead of
                        explicitly transforming each data point (which could be very high-dimensional or even infinite),
                        we only need to compute the kernel function between pairs of points.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-teal-100 dark:bg-teal-900 text-teal-600 dark:text-teal-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Decision Function</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        With the kernel trick, the SVM decision function becomes:
                        <span className="block mt-2 text-center font-mono">
                          f(x) = sign(Σ α<sub>i</sub>y<sub>i</sub>K(x<sub>i</sub>, x) + b)
                        </span>
                        where α<sub>i</sub> are the Lagrange multipliers, y<sub>i</sub> are the class labels, and
                        x<sub>i</sub> are the support vectors.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Common Kernel Functions</h4>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Linear Kernel</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      The simplest kernel function, equivalent to no transformation:
                      <span className="block mt-2 text-center font-mono">K(x, y) = x·y</span>
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Used when data is already linearly separable</li>
                      <li>Computationally efficient</li>
                      <li>Good for high-dimensional data like text</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Polynomial Kernel</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Maps data into a higher-dimensional space of polynomial features:
                      <span className="block mt-2 text-center font-mono">K(x, y) = (γx·y + r)<sup>d</sup></span>
                      where d is the degree, γ is the gamma parameter, and r is the coefficient.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Good for problems where all features are normalized</li>
                      <li>Degree d controls the flexibility of the decision boundary</li>
                      <li>Higher degrees can lead to overfitting</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Radial Basis Function (RBF) Kernel</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Also known as the Gaussian kernel, maps to an infinite-dimensional space:
                      <span className="block mt-2 text-center font-mono">
                        K(x, y) = exp(-γ||x - y||<sup>2</sup>)
                      </span>
                      where γ is the gamma parameter controlling the influence of each training example.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Most commonly used kernel for non-linear data</li>
                      <li>Works well in most cases when data is not too large</li>
                      <li>Gamma parameter controls the "reach" of each training example</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Sigmoid Kernel</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Creates a decision boundary similar to neural networks:
                      <span className="block mt-2 text-center font-mono">K(x, y) = tanh(γx·y + r)</span>
                      where γ is the gamma parameter and r is the coefficient.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Less commonly used than RBF or polynomial</li>
                      <li>Can behave like a linear kernel for certain parameters</li>
                      <li>Historically connected to neural networks</li>
                    </ul>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Choosing the Right Kernel</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  Selecting the appropriate kernel is crucial for SVM performance. Here are some guidelines:
                </p>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Start Simple:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Begin with a linear kernel, especially for high-dimensional data
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">RBF for Unknown Data Structure:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        If you're unsure about the data structure, RBF is often a good default
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Polynomial for Known Relationships:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Use polynomial kernels when you know the data has polynomial relationships
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Cross-Validation:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Always use cross-validation to compare different kernels and their parameters
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Domain Knowledge:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Incorporate domain knowledge about the problem when selecting a kernel
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Kernel SVM Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(kernelSVMCode, "kernel-svm-code")}
                    className="text-xs"
                  >
                    {copiedState === "kernel-svm-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{kernelSVMCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how to implement SVM with different kernel functions using scikit-learn. It
                    compares linear, polynomial, and RBF kernels on a non-linearly separable dataset and visualizes the
                    decision boundaries.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Kernel Transformation Visualization</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Kernel Transformation"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*Uw0wpvFDqrk4669XuY2_Dw.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The visualization shows:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Data that is not linearly separable in the original 2D space (left)</li>
                    <li>The same data transformed to a higher-dimensional space where it becomes linearly separable (right)</li>
                    <li>The kernel trick allows SVM to find this transformation implicitly without computing the actual coordinates</li>
                  </ul>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Comparison of Different Kernels</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Kernel Comparison"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_kernels_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">This visualization compares decision boundaries created by different kernel functions:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Linear kernel creates a straight-line boundary</li>
                    <li>Polynomial kernel creates a curved boundary with complexity based on the degree</li>
                    <li>RBF kernel creates a flexible boundary that can capture complex patterns</li>
                    <li>The choice of kernel significantly impacts the model's ability to separate classes</li>
                  </ul>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Effect of Gamma Parameter in RBF Kernel</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Gamma Parameter Effect"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_rbf_parameters_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The visualization shows how the gamma parameter affects the RBF kernel:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Small gamma values (left) create a smoother, more generalized decision boundary</li>
                    <li>Large gamma values (right) create a more complex boundary that closely fits the training data</li>
                    <li>Too small gamma can lead to underfitting, while too large gamma can lead to overfitting</li>
                    <li>The optimal gamma value depends on the specific dataset and should be tuned using cross-validation</li>
                  </ul>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-teal-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The kernel trick allows SVM to handle non-linearly separable data efficiently</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>It implicitly maps data to a higher-dimensional space without computing the actual coordinates</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Common kernels include linear, polynomial, RBF (Gaussian), and sigmoid</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The choice of kernel and its parameters significantly impacts model performance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Cross-validation should be used to select the best kernel and tune its parameters</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 4: Implementation with Scikit-Learn
    if (section === 4) {
      function copyToClipboard(svmClassificationCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-950 dark:to-teal-950 p-6 rounded-lg border border-green-100 dark:border-green-900">
            <h3 className="text-xl font-semibold text-green-800 dark:text-green-300 mb-3">
              Implementation with Scikit-Learn
            </h3>
            <p className="text-green-700 dark:text-green-300 leading-relaxed">
              Scikit-learn provides a powerful and easy-to-use implementation of Support Vector Machines. In this
              section, we'll explore how to implement SVM for both classification and regression tasks using
              scikit-learn, along with practical examples and best practices.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Implementation
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Results
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Scikit-Learn SVM Classes</h4>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>SVC</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Support Vector Classification for binary and multi-class classification tasks.
                    </p>
                    <div className="bg-muted/50 p-3 rounded text-sm">
                      <p className="font-mono">from sklearn.svm import SVC</p>
                      <p className="font-mono mt-1">model = SVC(kernel='rbf', C=1.0, gamma='scale')</p>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>LinearSVC</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Linear Support Vector Classification, optimized for linear kernels and large datasets.
                    </p>
                    <div className="bg-muted/50 p-3 rounded text-sm">
                      <p className="font-mono">from sklearn.svm import LinearSVC</p>
                      <p className="font-mono mt-1">model = LinearSVC(C=1.0, max_iter=1000)</p>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>SVR</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Support Vector Regression for continuous target variables.
                    </p>
                    <div className="bg-muted/50 p-3 rounded text-sm">
                      <p className="font-mono">from sklearn.svm import SVR</p>
                      <p className="font-mono mt-1">model = SVR(kernel='rbf', C=1.0, epsilon=0.1)</p>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>LinearSVR</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Linear Support Vector Regression, optimized for linear kernels and large datasets.
                    </p>
                    <div className="bg-muted/50 p-3 rounded text-sm">
                      <p className="font-mono">from sklearn.svm import LinearSVR</p>
                      <p className="font-mono mt-1">model = LinearSVR(C=1.0, epsilon=0.1)</p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Important Parameters</h4>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      C
                    </Badge>
                    <div>
                      <p className="text-sm">Regularization parameter (default: 1.0)</p>
                      <p className="text-xs text-muted-foreground">
                        Controls the trade-off between achieving a low training error and a low testing error
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      kernel
                    </Badge>
                    <div>
                      <p className="text-sm">Kernel type to be used (default: 'rbf')</p>
                      <p className="text-xs text-muted-foreground">
                        Options: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      gamma
                    </Badge>
                    <div>
                      <p className="text-sm">Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (default: 'scale')</p>
                      <p className="text-xs text-muted-foreground">
                        Options: 'scale', 'auto', or a float value
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      degree
                    </Badge>
                    <div>
                      <p className="text-sm">Degree of the polynomial kernel function (default: 3)</p>
                      <p className="text-xs text-muted-foreground">
                        Only significant for 'poly' kernel
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      probability
                    </Badge>
                    <div>
                      <p className="text-sm">Whether to enable probability estimates (default: False)</p>
                      <p className="text-xs text-muted-foreground">
                        Enables predict_proba() method but slows down fitting
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      class_weight
                    </Badge>
                    <div>
                      <p className="text-sm">Class weights for imbalanced datasets (default: None)</p>
                      <p className="text-xs text-muted-foreground">
                        Options: 'balanced', dictionary, or None
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      epsilon
                    </Badge>
                    <div>
                      <p className="text-sm">Epsilon in the epsilon-SVR model (default: 0.1)</p>
                      <p className="text-xs text-muted-foreground">
                        Specifies the epsilon-tube within which no penalty is associated in the training loss function
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Implementation Steps</h4>
                <ol className="space-y-3 list-decimal pl-5">
                  <li>Import the necessary libraries and SVM classes</li>
                  <li>Load and prepare your dataset</li>
                  <li>Split the data into training and testing sets</li>
                  <li>Scale features (important for SVM performance)</li>
                  <li>Create and configure the SVM model</li>
                  <li>Train the model using the fit() method</li>
                  <li>Make predictions using the predict() method</li>
                  <li>Evaluate the model's performance</li>
                </ol>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Best Practices</h4>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Feature Scaling:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Always scale your features (e.g., using StandardScaler or MinMaxScaler) as SVM is sensitive to
                        feature scales
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Parameter Tuning:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Use cross-validation and grid search to find optimal values for C, gamma, and other parameters
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Handling Imbalanced Data:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Use the class_weight parameter or resampling techniques for imbalanced datasets
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Large Datasets:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        For large datasets, consider using LinearSVC or SGDClassifier with 'hinge' loss instead of SVC
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Kernel Selection:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Start with a linear kernel for high-dimensional data and RBF for low-dimensional data
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">SVM Classification Example</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(svmClassificationCode, "svm-classification-code")}
                    className="text-xs"
                  >
                    {copiedState === "svm-classification-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{svmClassificationCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates a complete SVM classification workflow using scikit-learn. It includes data
                    loading, preprocessing, model training, prediction, and evaluation with various metrics.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">SVM Regression Example</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(svmRegressionCode, "svm-regression-code")}
                    className="text-xs"
                  >
                    {copiedState === "svm-regression-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{svmRegressionCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code shows how to implement Support Vector Regression (SVR) using scikit-learn. It includes
                    data preparation, model training, prediction, and evaluation with regression metrics.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Classification Results</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=400"
                      alt="Confusion Matrix"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src =
                          "https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_001.png"
                      }}
                    />
                  </div>
                  <div className="flex flex-col justify-between">
                    <div>
                      <h5 className="font-medium mb-2">Confusion Matrix</h5>
                      <p className="text-sm text-muted-foreground mb-4">
                        The confusion matrix shows how well our SVM model is performing for each class. The diagonal
                        elements represent correct predictions, while off-diagonal elements are misclassifications.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h5 className="font-medium mb-2">Performance Metrics</h5>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Accuracy:</span>
                          <Badge variant="outline">96.7%</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Precision:</span>
                          <Badge variant="outline">97.2%</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Recall:</span>
                          <Badge variant="outline">96.7%</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">F1 Score:</span>
                          <Badge variant="outline">96.8%</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Regression Results</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=400"
                      alt="Predicted vs Actual Values"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src =
                          "https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_regression_001.png"
                      }}
                    />
                  </div>
                  <div className="flex flex-col justify-between">
                    <div>
                      <h5 className="font-medium mb-2">Predicted vs Actual Values</h5>
                      <p className="text-sm text-muted-foreground mb-4">
                        The scatter plot shows the relationship between predicted and actual values. Points closer to
                        the diagonal line indicate better predictions.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h5 className="font-medium mb-2">Performance Metrics</h5>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">R² Score:</span>
                          <Badge variant="outline">0.85</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Mean Absolute Error:</span>
                          <Badge variant="outline">0.42</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Mean Squared Error:</span>
                          <Badge variant="outline">0.31</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Root Mean Squared Error:</span>
                          <Badge variant="outline">0.56</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">SVM Decision Boundary</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="SVM Decision Boundary"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p>
                    The visualization shows the decision boundaries created by an SVM model with an RBF kernel on the
                    Iris dataset. Different colors represent different classes, and the decision boundaries show where
                    the model transitions from predicting one class to another.
                  </p>
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
                <span>Scikit-learn provides efficient implementations of SVM for both classification and regression</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Feature scaling is crucial for SVM performance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The choice of kernel and parameters significantly impacts model performance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>For large datasets, LinearSVC is more efficient than SVC with a linear kernel</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Cross-validation should be used to find optimal parameter values</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 5: Hyperparameter Tuning
    if (section === 5) {
      function copyToClipboard(gridSearchSVMCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950 dark:to-orange-950 p-6 rounded-lg border border-amber-100 dark:border-amber-900">
            <h3 className="text-xl font-semibold text-amber-800 dark:text-amber-300 mb-3">Hyperparameter Tuning</h3>
            <p className="text-amber-700 dark:text-amber-300 leading-relaxed">
              Tuning hyperparameters is crucial for optimizing SVM performance. The right combination of parameters can
              significantly improve model accuracy and generalization. In this section, we'll explore systematic
              approaches to find the best hyperparameters for your SVM models.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Key Parameters
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Results
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Key Hyperparameters to Tune</h4>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>C (Regularization Parameter)</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Controls the trade-off between achieving a low training error and a low testing error.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Small C: Prioritizes a wider margin, allows more misclassifications</li>
                      <li>Large C: Prioritizes correct classification, creates a narrower margin</li>
                      <li>Typical range: 0.1 to 100 (logarithmic scale)</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>gamma (Kernel Coefficient)</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Defines how far the influence of a single training example reaches for 'rbf', 'poly', and
                      'sigmoid' kernels. a single training example reaches for 'rbf', 'poly', and 'sigmoid' kernels.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Small gamma: Far reach, smoother decision boundary</li>
                      <li>Large gamma: Close reach, more complex decision boundary</li>
                      <li>Typical range: 0.001 to 10 (logarithmic scale)</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>kernel</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Specifies the kernel type to be used in the algorithm.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>'linear': Linear kernel, fastest for linearly separable data</li>
                      <li>'poly': Polynomial kernel, good for curved boundaries</li>
                      <li>'rbf': Radial Basis Function, most versatile for complex patterns</li>
                      <li>'sigmoid': Creates neural network-like decision boundaries</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>degree</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Degree of the polynomial kernel function.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Only relevant when kernel='poly'</li>
                      <li>Higher degree creates more complex, curved boundaries</li>
                      <li>Typical range: 2 to 5</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>epsilon</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      For regression models (SVR), defines the epsilon-tube within which no penalty is associated.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Controls the width of the insensitive region</li>
                      <li>Smaller values create more complex models</li>
                      <li>Typical range: 0.01 to 1.0</li>
                    </ul>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Tuning Methods</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4 bg-amber-50 dark:bg-amber-950">
                    <h5 className="font-medium mb-2">Grid Search</h5>
                    <p className="text-sm text-muted-foreground">
                      Exhaustively searches through a specified parameter grid, evaluating all possible combinations.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5 mt-2">
                      <li>Thorough but computationally expensive</li>
                      <li>Good for small parameter spaces</li>
                      <li>Guaranteed to find the best combination in the grid</li>
                    </ul>
                  </div>
                  <div className="border rounded-lg p-4 bg-orange-50 dark:bg-orange-950">
                    <h5 className="font-medium mb-2">Random Search</h5>
                    <p className="text-sm text-muted-foreground">
                      Randomly samples parameter combinations from specified distributions.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5 mt-2">
                      <li>More efficient for large parameter spaces</li>
                      <li>Can explore more values per parameter</li>
                      <li>Often finds good solutions with fewer iterations</li>
                    </ul>
                  </div>
                </div>
                <div className="border rounded-lg p-4 mt-4 bg-red-50 dark:bg-red-950">
                  <h5 className="font-medium mb-2">Bayesian Optimization</h5>
                  <p className="text-sm text-muted-foreground">
                    Uses probabilistic models to guide the search for optimal parameters.
                  </p>
                  <ul className="text-sm space-y-1 list-disc pl-5 mt-2">
                    <li>More efficient than grid or random search</li>
                    <li>Learns from previous evaluations</li>
                    <li>Particularly useful for expensive-to-evaluate models</li>
                    <li>Implemented in libraries like scikit-optimize or Optuna</li>
                  </ul>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Cross-Validation</h4>
                <p className="text-sm text-muted-foreground mb-4">
                  When tuning hyperparameters, it's important to use cross-validation to get reliable performance
                  estimates.
                </p>
                <div className="bg-muted/50 p-4 rounded-lg">
                  <h5 className="font-medium mb-2">K-Fold Cross-Validation</h5>
                  <ol className="text-sm space-y-2 list-decimal pl-5">
                    <li>Split the data into K folds (typically 5 or 10)</li>
                    <li>
                      For each parameter combination:
                      <ul className="list-disc pl-5 mt-1">
                        <li>Train on K-1 folds</li>
                        <li>Validate on the remaining fold</li>
                        <li>Repeat for all K folds</li>
                        <li>Average the performance metrics</li>
                      </ul>
                    </li>
                    <li>Select the parameter combination with the best average performance</li>
                    <li>Retrain on the entire training set with the best parameters</li>
                    <li>Evaluate on the test set (which was not used in tuning)</li>
                  </ol>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Grid Search CV</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(gridSearchSVMCode, "grid-search-svm-code")}
                    className="text-xs"
                  >
                    {copiedState === "grid-search-svm-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{gridSearchSVMCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how to use GridSearchCV to find the optimal hyperparameters for an SVM
                    model. It exhaustively searches through all combinations of the specified parameter values.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Random Search CV</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(randomSearchSVMCode, "random-search-svm-code")}
                    className="text-xs"
                  >
                    {copiedState === "random-search-svm-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{randomSearchSVMCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code shows how to use RandomizedSearchCV, which is more efficient for large parameter spaces.
                    It randomly samples parameter combinations rather than trying all possibilities.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Parameter Importance</h4>
                <div className="h-80">
                  <ChartContainer>
                    <ChartTitle>Hyperparameter Importance</ChartTitle>
                    <ChartBar
                      data={[
                        { parameter: "C", importance: 0.42, color: "#f59e0b" },
                        { parameter: "gamma", importance: 0.35, color: "#f59e0b" },
                        { parameter: "kernel", importance: 0.15, color: "#f59e0b" },
                        { parameter: "degree", importance: 0.08, color: "#f59e0b" },
                      ]}
                    >
                      {(data) => (
                        <>
                          <ChartBarItem
                            data={data}
                            valueAccessor={(d) => d.importance}
                            categoryAccessor={(d) => d.parameter}
                            style={{ fill: (d: { color: string }) => d.color }}
                          />
                        </>
                      )}
                    </ChartBar>
                    <ChartTooltip>
                      {({ point }) => (
                        <ChartTooltipContent>
                          <ChartTooltipItem label="Parameter" value={point.data.parameter} />
                          <ChartTooltipItem label="Importance" value={point.data.importance.toFixed(2)} />
                        </ChartTooltipContent>
                      )}
                    </ChartTooltip>
                  </ChartContainer>
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  <p>
                    The chart shows the relative importance of different hyperparameters in determining model
                    performance. This can help focus tuning efforts on the most influential parameters.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Performance Comparison</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=300&width=500"
                      alt="Hyperparameter Tuning Results"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src =
                          "https://scikit-learn.org/stable/_images/sphx_glr_plot_rbf_parameters_001.png"
                      }}
                    />
                  </div>
                  <div className="flex flex-col justify-between">
                    <div>
                      <h5 className="font-medium mb-2">Tuning Results</h5>
                      <p className="text-sm text-muted-foreground mb-4">
                        The plot shows how model performance varies with different C and gamma values for an RBF kernel.
                        Darker regions indicate better performance, helping identify the optimal parameter combination.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg">
                      <h5 className="font-medium mb-2">Best Parameters</h5>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between items-center">
                          <span>kernel:</span>
                          <Badge variant="outline">'rbf'</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>C:</span>
                          <Badge variant="outline">10.0</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>gamma:</span>
                          <Badge variant="outline">0.1</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span>Performance Improvement:</span>
                          <Badge variant="outline">+4.5%</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Learning Curves</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="Learning Curves"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src =
                        "https://scikit-learn.org/stable/_images/sphx_glr_plot_learning_curve_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p>
                    Learning curves show how model performance changes with increasing training set size. They help
                    diagnose overfitting or underfitting and determine if more data would be beneficial. The gap
                    between training and validation scores indicates the degree of overfitting.
                  </p>
                </div>
              </Card>
            </TabsContent>
          </Tabs>

          <Card className="p-5 mt-6 border-l-4 border-l-amber-500">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-lg">Key Takeaways</h4>
            </div>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Hyperparameter tuning is crucial for optimizing SVM performance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>C and gamma are the most important parameters to tune for RBF kernel</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Grid search is thorough but slow; random search is more efficient for large parameter spaces</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Cross-validation is essential for reliable performance estimation</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Learning curves help diagnose overfitting or underfitting and guide parameter selection</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 6: Visualization & Interpretation
    if (section === 6) {
      function copyToClipboard(decisionBoundaryCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-950 dark:to-blue-950 p-6 rounded-lg border border-indigo-100 dark:border-indigo-900">
            <h3 className="text-xl font-semibold text-indigo-800 dark:text-indigo-300 mb-3">
              Visualization & Interpretation
            </h3>
            <p className="text-indigo-700 dark:text-indigo-300 leading-relaxed">
              Visualizing and interpreting SVM models helps us understand how they make decisions and identify areas for
              improvement. While SVMs are not as inherently interpretable as some other algorithms, several techniques
              can provide insights into their behavior and performance.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Interpretation
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Example
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Visualizations
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Interpreting SVM Models</h4>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Support Vectors</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Support vectors are the data points that lie closest to the decision boundary. They are the most
                        influential points in defining the hyperplane and margin. Examining support vectors can provide
                        insights into which examples are most critical for the model's decisions.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Decision Function</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        The decision function (w·x + b) gives the signed distance of a point from the decision boundary.
                        The magnitude indicates how confidently the model classifies a point, with larger absolute
                        values suggesting higher confidence.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Feature Weights (Linear SVM)</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        For linear SVMs, the coefficients (w) represent the importance of each feature in the decision
                        boundary. Features with larger absolute coefficient values have a greater impact on the
                        classification decision.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Decision Boundaries</h5>
                      <p className="text-sm text-muted-foreground mt-1">
                        Visualizing decision boundaries helps understand how the model separates different classes in
                        the feature space. This is particularly useful for 2D or 3D data, or when using dimensionality
                        reduction techniques.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Visualization Techniques</h4>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Decision Boundary Plots</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Visualize how the model separates different classes in the feature space.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Works directly for 2D data</li>
                      <li>For higher dimensions, use dimensionality reduction (PCA, t-SNE) first</li>
                      <li>Shows the margin and support vectors</li>
                      <li>Helps understand the impact of different kernels and parameters</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Feature Importance (Linear SVM)</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      For linear SVMs, visualize the coefficients to understand feature importance.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Larger absolute values indicate more influential features</li>
                      <li>Sign indicates the direction of influence</li>
                      <li>Can help with feature selection</li>
                      <li>Not directly applicable to non-linear kernels</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>Confusion Matrix</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Shows the model's performance across different classes.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>Identifies which classes are confused with each other</li>
                      <li>Helps detect class imbalance issues</li>
                      <li>Provides insights for targeted improvements</li>
                      <li>Useful for multi-class classification</li>
                    </ul>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h5 className="font-medium mb-2 flex items-center gap-2">
                      <Badge>ROC and Precision-Recall Curves</Badge>
                    </h5>
                    <p className="text-sm text-muted-foreground mb-2">
                      Evaluate model performance across different threshold values.
                    </p>
                    <ul className="text-sm space-y-1 list-disc pl-5">
                      <li>ROC curve: True Positive Rate vs. False Positive Rate</li>
                      <li>Precision-Recall curve: Precision vs. Recall</li>
                      <li>AUC (Area Under Curve) provides a single performance metric</li>
                      <li>Particularly useful for imbalanced datasets</li>
                    </ul>
                  </div>
                </div>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Advanced Interpretation Methods</h4>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Permutation Importance:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Measures the decrease in model performance when a feature is randomly shuffled
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">Partial Dependence Plots:</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Show how the model's predictions change as a function of a single feature
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">SHAP (SHapley Additive exPlanations):</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Provides consistent, locally accurate feature importance values based on game theory
                      </span>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <ArrowRight className="h-4 w-4 text-primary mt-1" />
                    <div>
                      <span className="font-medium">LIME (Local Interpretable Model-agnostic Explanations):</span>
                      <span className="text-sm text-muted-foreground ml-1">
                        Explains individual predictions by approximating the model locally with an interpretable one
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Visualizing Decision Boundaries</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(decisionBoundaryCode, "decision-boundary-code")}
                    className="text-xs"
                  >
                    {copiedState === "decision-boundary-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{decisionBoundaryCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates how to visualize decision boundaries for SVM models with different kernels.
                    It creates a mesh grid over the feature space and predicts the class for each point, then plots the
                    results as colored regions.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Feature Importance and Model Evaluation</h4>
                  <Button
                    onClick={() => copyToClipboard(featureImportanceSVMCode, "feature-importance-svm-code")}
                    size="sm"
                    className="text-xs"
                  >
                    {copiedState === "feature-importance-svm-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{featureImportanceSVMCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code shows how to extract and visualize feature importance from a linear SVM model, create a
                    confusion matrix, and generate ROC and precision-recall curves for model evaluation.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Decision Boundaries with Different Kernels</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="SVM Decision Boundaries"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_kernels_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p className="mb-2">The visualization shows decision boundaries created by different kernel functions:</p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Linear kernel creates a straight-line boundary</li>
                    <li>Polynomial kernel creates a curved boundary with complexity based on the degree</li>
                    <li>RBF kernel creates a flexible boundary that can capture complex patterns</li>
                    <li>Support vectors are highlighted as the points that define the margin</li>
                  </ul>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Feature Importance (Linear SVM)</h4>
                <div className="h-80">
                  <ChartContainer>
                    <ChartTitle>Feature Importance</ChartTitle>
                    <ChartBar
                      data={[
                        { feature: "Feature A", importance: 0.85, color: "#6366f1" },
                        { feature: "Feature B", importance: -0.62, color: "#f43f5e" },
                        { feature: "Feature C", importance: 0.45, color: "#6366f1" },
                        { feature: "Feature D", importance: -0.38, color: "#f43f5e" },
                        { feature: "Feature E", importance: 0.22, color: "#6366f1" },
                      ]}
                    >
                      {(data) => (
                        <>
                          <ChartBarItem
                            data={data}
                            valueAccessor={(d) => d.importance}
                            categoryAccessor={(d) => d.feature}
                            style={{ fill: (d: { color: string }) => d.color }}
                          />
                        </>
                      )}
                    </ChartBar>
                    <ChartTooltip>
                      {({ point }) => (
                        <ChartTooltipContent>
                          <ChartTooltipItem label="Feature" value={point.data.feature} />
                          <ChartTooltipItem label="Importance" value={point.data.importance.toFixed(2)} />
                        </ChartTooltipContent>
                      )}
                    </ChartTooltip>
                  </ChartContainer>
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  <p>
                    The chart shows the feature importance (coefficients) from a linear SVM model. Positive values
                    (blue) indicate features that contribute to the positive class, while negative values (red)
                    contribute to the negative class. The magnitude indicates the strength of the contribution.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">Confusion Matrix</h4>
                <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=400"
                    alt="Confusion Matrix"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src =
                        "https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p>
                    The confusion matrix shows the model's performance across different classes. Each row represents the
                    actual class, and each column represents the predicted class. The diagonal elements show correct
                    predictions, while off-diagonal elements show misclassifications. This helps identify which classes
                    are being confused with each other.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <h4 className="font-medium text-lg mb-3">ROC Curve</h4>
                <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=400&width=800"
                    alt="ROC Curve"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src =
                        "https://scikit-learn.org/stable/_images/sphx_glr_plot_roc_001.png"
                    }}
                  />
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p>
                    The ROC (Receiver Operating Characteristic) curve shows the trade-off between the true positive rate
                    and false positive rate at different classification thresholds. The area under the curve (AUC)
                    provides a single measure of model performance, with higher values indicating better performance.
                    This is particularly useful for evaluating binary classification models and comparing different
                    models.
                  </p>
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
                <span>Visualizing decision boundaries helps understand how SVM separates different classes</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>For linear SVMs, feature coefficients provide insights into feature importance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Confusion matrices help identify which classes are being confused with each other</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>ROC and precision-recall curves evaluate model performance across different thresholds</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Advanced techniques like SHAP and LIME can provide more detailed model interpretations</span>
              </li>
            </ul>
          </Card>

          <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-950 dark:to-blue-950 p-6 rounded-lg border border-indigo-100 dark:border-indigo-900 mt-6">
            <h3 className="text-xl font-semibold text-indigo-800 dark:text-indigo-300 mb-3">Conclusion</h3>
            <p className="text-indigo-700 dark:text-indigo-300 leading-relaxed mb-4">
              Support Vector Machines are powerful and versatile algorithms for classification and regression tasks.
              They work by finding the optimal hyperplane that maximizes the margin between different classes, and can
              handle both linearly and non-linearly separable data through the use of kernel functions.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">What We've Learned</h4>
                <ul className="text-sm space-y-1 list-disc pl-5">
                  <li>The mathematical foundations of SVM</li>
                  <li>How linear SVM works for linearly separable data</li>
                  <li>The kernel trick for handling non-linear data</li>
                  <li>Implementation with scikit-learn for both classification and regression</li>
                  <li>Hyperparameter tuning to optimize model performance</li>
                  <li>Visualization and interpretation techniques</li>
                </ul>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Next Steps</h4>
                <ul className="text-sm space-y-1 list-disc pl-5">
                  <li>Explore other kernel functions for specific applications</li>
                  <li>Combine SVM with feature selection techniques</li>
                  <li>Apply SVM to real-world datasets</li>
                  <li>Compare SVM with other classification algorithms</li>
                  <li>Investigate advanced interpretation methods</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="flex justify-center mt-8">
            <Button size="lg" className="gap-2">
              <ArrowRight className="h-4 w-4" />
              Continue to Practice Exercises
            </Button>
          </div>
        </div>
      )
    }

    // Default return if section is not found
    return (
      <div className="py-8 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <BookOpen className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-xl font-medium mb-2">Section Content Coming Soon</h3>
        <p className="text-muted-foreground max-w-md mx-auto">
          We're currently developing content for this section of the SVM tutorial. Check back soon!
        </p>
      </div>
    )
  }

// Code examples as constants
const svmMathCode = `import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# Disable output from CVXOPT solver
solvers.options['show_progress'] = False

def linear_kernel(x1, x2):
    """
    Compute the linear kernel between two vectors
    """
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3):
    """
    Compute the polynomial kernel between two vectors
    """
    return (np.dot(x1, x2) + 1) ** degree

def rbf_kernel(x1, x2, gamma=1.0):
    """
    Compute the RBF (Gaussian) kernel between two vectors
    """
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

class SVM:
    def __init__(self, kernel=linear_kernel, C=1.0):
        """
        Initialize SVM with kernel function and regularization parameter
        """
        self.kernel = kernel
        self.C = C
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0
    
    def fit(self, X, y):
        """
        Train the SVM model using quadratic programming
        """
        n_samples, n_features = X.shape
        
        # Compute the kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])
        
        # Set up the quadratic programming problem
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        
        # Inequality constraints: 0 <= alpha_i <= C
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        
        # Equality constraint: sum(alpha_i * y_i) = 0
        A = matrix(y.astype(float)).trans()
        b = matrix(0.0)
        
        # Solve the quadratic programming problem
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x']).flatten()
        
        # Find support vectors (alphas > 0)
        sv_indices = alphas > 1e-5
        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        
        # Compute bias term
        self.b = 0
        for i in range(len(self.alphas)):
            self.b += self.support_vector_labels[i]
            self.b -= np.sum(self.alphas * self.support_vector_labels * 
                             K[sv_indices, sv_indices[i]])
        self.b /= len(self.alphas)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        y_pred = np.zeros(len(X))
        
        for i in range(len(X)):
            s = 0
            for alpha, sv_y, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                s += alpha * sv_y * self.kernel(X[i], sv)
            y_pred[i] = s + self.b
        
        return np.sign(y_pred)
    
    def decision_function(self, X):
        """
        Compute the decision function for samples in X
        """
        y_decision = np.zeros(len(X))
        
        for i in range(len(X)):
            s = 0
            for alpha, sv_y, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                s += alpha * sv_y * self.kernel(X[i], sv)
            y_decision[i] = s + self.b
        
        return y_decision

# Example usage
if __name__ == "__main__":
    # Generate a simple linearly separable dataset
    np.random.seed(42)
    X_pos = np.random.randn(50, 2) + np.array([2, 2])
    X_neg = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(50), -np.ones(50)))
    
    # Train SVM with linear kernel
    svm = SVM(kernel=linear_kernel, C=1.0)
    svm.fit(X, y)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c='b', marker='+', label='Positive')
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c='r', marker='o', label='Negative')
    
    # Plot the support vectors
    plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                s=100, facecolors='none', edgecolors='g', label='Support Vectors')
    
    # Create a mesh grid to visualize the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Compute the decision function for all mesh grid points
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i, j] = svm.decision_function(np.array([[xx[i, j], yy[i, j]]]))
    
    # Plot the decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['r', 'k', 'b'], 
                linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary with Support Vectors')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print information about the model
    print(f"Number of support vectors: {len(svm.support_vectors)}")
    print(f"Bias term (b): {svm.b:.4f}")
    
    # Calculate the margin
    w = np.zeros(2)
    for i in range(len(svm.alphas)):
        w += svm.alphas[i] * svm.support_vector_labels[i] * svm.support_vectors[i]
    margin = 2 / np.linalg.norm(w)
    print(f"Margin: {margin:.4f}")`

const linearSVMCode = `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Generate a synthetic dataset
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the linear SVM model
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Visualize the decision boundary
def plot_decision_boundary(X, y, model, scaler):
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Scale the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    
    # Predict the class for each point in the mesh grid
    Z = model.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k')
    
    # Plot the support vectors
    plt.scatter(X[model.support_], y[model.support_], s=100, 
                facecolors='none', edgecolors='k', linewidths=2, 
                label='Support Vectors')
    
    # Plot the decision boundary and margins
    w = model.coef_[0]
    b = model.intercept_[0]
    
    # Decision boundary: w[0]*x + w[1]*y + b = 0
    # y = -(w[0]*x + b) / w[1]
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    
    # Plot the decision boundary
    x_boundary = np.array([x_min, x_max])
    y_boundary = slope * x_boundary + intercept
    plt.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Decision Boundary')
    
    # Plot the margins
    margin = 1 / np.sqrt(np.sum(w ** 2))
    y_margin_above = slope * x_boundary + intercept + margin
    y_margin_below = slope * x_boundary + intercept - margin
    plt.plot(x_boundary, y_margin_above, 'k--', linewidth=1, label='Margin')
    plt.plot(x_boundary, y_margin_below, 'k--', linewidth=1)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear SVM Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualize the decision boundary
plot_decision_boundary(X, y, svm, scaler)

# Print information about the model
print(f"Number of support vectors: {svm.n_support_}")
print(f"Coefficients (w): {svm.coef_[0]}")
print(f"Intercept (b): {svm.intercept_[0]}")

# Calculate the margin
w_norm = np.linalg.norm(svm.coef_[0])
margin = 2 / w_norm
print(f"Margin: {margin:.4f}")

# Predict probabilities (requires probability=True)
svm_prob = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_prob.fit(X_train_scaled, y_train)
probabilities = svm_prob.predict_proba(X_test_scaled)
print("\\nProbabilities for the first 5 test examples:")
for i in range(5):
    print(f"Example {i+1}: Class 0: {probabilities[i, 0]:.4f}, Class 1: {probabilities[i, 1]:.4f}")`

const kernelSVMCode = `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate a non-linearly separable dataset (concentric circles)
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train SVM models with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
models = {}
accuracies = {}

for kernel in kernels:
    # Create and train the model
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the model and accuracy
    models[kernel] = svm
    accuracies[kernel] = accuracy
    
    print(f"{kernel.capitalize()} Kernel Accuracy: {accuracy:.4f}")

# Visualize the decision boundaries
def plot_decision_boundaries(X, y, models, scaler):
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Scale the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    # Plot each kernel's decision boundary
    for i, (kernel, model) in enumerate(models.items()):
        # Predict the class for each point in the mesh grid
        Z = model.predict(mesh_points_scaled)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        axs[i].contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        
        # Plot the training points
        scatter = axs[i].scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k')
        
        # Plot the support vectors
        axs[i].scatter(X[model.support_], y[model.support_], s=100, 
                    facecolors='none', edgecolors='k', linewidths=2)
        
        axs[i].set_xlabel('Feature 1')
        axs[i].set_ylabel('Feature 2')
        axs[i].set_title(f'{kernel.capitalize()} Kernel (Accuracy: {accuracies[kernel]:.4f})')
        axs[i].legend(*scatter.legend_elements(), title="Classes")
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# Visualize the decision boundaries
plot_decision_boundaries(X, y, models, scaler)

# Explore the effect of gamma parameter on RBF kernel
gammas = [0.01, 0.1, 1, 10, 100]
rbf_models = {}
rbf_accuracies = {}

for gamma in gammas:
    # Create and train the model
    svm = SVC(kernel='rbf', gamma=gamma, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the model and accuracy
    rbf_models[gamma] = svm
    rbf_accuracies[gamma] = accuracy
    
    print(f"RBF Kernel (gamma={gamma}) Accuracy: {accuracy:.4f}")

# Visualize the effect of gamma
def plot_gamma_effect(X, y, models, scaler,
`