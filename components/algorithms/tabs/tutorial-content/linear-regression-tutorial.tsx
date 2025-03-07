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
} from "lucide-react"

interface LinearRegressionTutorialProps {
  section: number
  onCopy: (text: string, id: string) => void
  copied: string | null
}

export function LinearRegressionTutorial({ section, onCopy, copied }: LinearRegressionTutorialProps) {
  const [activeTab, setActiveTab] = useState("explanation")

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
                  onClick={() => onCopy(simpleLRCode, "simple-lr-code")}
                  className="text-xs"
                >
                  {copied === "simple-lr-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                <code>{simpleLRCode}</code>
              </pre>
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
                  onClick={() => onCopy(multipleLRCode, "multiple-lr-code")}
                  className="text-xs"
                >
                  {copied === "multiple-lr-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                <code>{multipleLRCode}</code>
              </pre>
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
                    e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*ndDZ2HXWPWSVhH7ZQ4hoiw.png"
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
                  onClick={() => onCopy(evaluationCode, "evaluation-code")}
                  className="text-xs"
                >
                  {copied === "evaluation-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                <code>{evaluationCode}</code>
              </pre>
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
                  onClick={() => onCopy(regularizationCode, "regularization-code")}
                  className="text-xs"
                >
                  {copied === "regularization-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                <code>{regularizationCode}</code>
              </pre>
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
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="border rounded-lg p-3">
              <h5 className="font-medium mb-2 text-sm">Data Preprocessing</h5>
              <p className="text-xs text-muted-foreground">
                Always check for and handle missing values, outliers, and feature scaling before building your model.
              </p>
            </div>
            <div className="border rounded-lg p-3">
              <h5 className="font-medium mb-2 text-sm">Feature Selection</h5>
              <p className="text-xs text-muted-foreground">
                Not all features are useful. Use techniques like Lasso or feature importance to select the most relevant
                ones.
              </p>
            </div>
            <div className="border rounded-lg p-3">
              <h5 className="font-medium mb-2 text-sm">Assumption Checking</h5>
              <p className="text-xs text-muted-foreground">
                Verify linear regression assumptions like linearity, independence, and homoscedasticity for valid
                results.
              </p>
            </div>
            <div className="border rounded-lg p-3">
              <h5 className="font-medium mb-2 text-sm">Hyperparameter Tuning</h5>
              <p className="text-xs text-muted-foreground">
                Use cross-validation to find the optimal regularization strength and other parameters for your model.
              </p>
            </div>
          </div>
        </Card>

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
        We're currently developing content for this section of the Linear Regression tutorial. Check back soon!
      </p>
    </div>
  )
}

// Code examples as constants
const simpleLRCode = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create sample data
hours_studied = np.array([1, 3, 4, 5, 7, 8, 10]).reshape(-1, 1)
exam_scores = np.array([18, 31, 40, 55, 71, 85, 97])

# Visualize data
plt.figure(figsize=(10, 6))
plt.scatter(hours_studied, exam_scores, color='blue', label='Data points')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours vs Exam Score')
plt.grid(True, alpha=0.3)

# Create and fit the model
model = LinearRegression()
model.fit(hours_studied, exam_scores)

# Get coefficients
slope = model.coef_[0]
intercept = model.intercept_
print(f"Equation: Score = {slope:.2f} × Hours + {intercept:.2f}")

# Creating the prediction line
x_range = np.linspace(0, 12, 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# Plot data and prediction line
plt.plot(x_range, y_pred, color='red', label='Regression line')
plt.legend()
plt.show()

# Predict score for 6 hours of studying
new_hours = np.array([[6]])
predicted_score = model.predict(new_hours)
print(f"Predicted score for 6 hours of studying: {predicted_score[0]:.2f}")`

const multipleLRCode = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Sample data with multiple features: hours studied, hours slept
X = np.array([
    [1, 5],  # [hours_studied, hours_slept]
    [3, 7],
    [4, 8],
    [5, 5],
    [7, 9],
    [8, 6],
    [10, 7]
])
y = np.array([18, 31, 40, 55, 71, 85, 97])  # exam scores

# Create and fit model
multi_model = LinearRegression()
multi_model.fit(X, y)

# Print coefficients
print(f"Equation: Score = {multi_model.coef_[0]:.2f} × Hours_Studied + "
      f"{multi_model.coef_[1]:.2f} × Hours_Slept + {multi_model.intercept_:.2f}")

# Normalize features (important for multiple regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit model with normalized features
normalized_model = LinearRegression()
normalized_model.fit(X_scaled, y)

# Print coefficients for normalized model
print("\\nWith normalized features:")
print(f"Equation: Score = {normalized_model.coef_[0]:.2f} × Hours_Studied_Scaled + "
      f"{normalized_model.coef_[1]:.2f} × Hours_Slept_Scaled + {normalized_model.intercept_:.2f}")

# Make predictions with new data
new_data = np.array([[6, 8]])  # 6 hours studied, 8 hours slept
new_data_scaled = scaler.transform(new_data)

# Predict with both models
pred_regular = multi_model.predict(new_data)
pred_normalized = normalized_model.predict(new_data_scaled)

print(f"\\nPredicted score with regular model: {pred_regular[0]:.2f}")
print(f"Predicted score with normalized model: {pred_normalized[0]:.2f}")`

const evaluationCode = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create sample data
X = np.array([1, 3, 4, 5, 7, 8, 10, 12, 13, 14, 15, 16, 18, 19, 21]).reshape(-1, 1)
y = np.array([18, 31, 40, 55, 71, 85, 97, 106, 114, 119, 130, 135, 142, 153, 158])

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

# Plot residuals
plt.figure(figsize=(12, 5))

# Training set residuals
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, y_train - y_train_pred)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Training Set Residual Plot')
plt.grid(True, alpha=0.3)

# Test set residuals
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, y_test - y_test_pred)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Test Set Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()`

const regularizationCode = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# Generate synthetic data with some noise
X, y = make_regression(n_samples=100, n_features=20, noise=20, random_state=42)

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

# Plot coefficients
plt.figure(figsize=(12, 8))
for i, (name, metrics) in enumerate(results.items()):
    plt.subplot(2, 2, i+1)
    plt.bar(range(len(metrics['coef'])), metrics['coef'])
    plt.title(f"{name} - Coefficients")
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

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
    print(f"Alpha = {alpha}: R² = {metrics['test_r2']:.4f}, RMSE = {metrics['test_rmse']:.4f}")`

