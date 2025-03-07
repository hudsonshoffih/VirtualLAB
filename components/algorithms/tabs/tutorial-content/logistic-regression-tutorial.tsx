"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  TrendingUp,
  BookOpen,
  Code,
  BarChart,
  Lightbulb,
  CheckCircle,
  ArrowRight,
  Sigma,
  GitBranch,
  Activity,
  Target,
} from "lucide-react"

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
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                <code>{sigmoidCode}</code>
              </pre>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Sigmoid Function Visualization</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="/sigmoid-function.png"
                  alt="Sigmoid Function"
                  className="max-w-full h-auto"
                  onError={(e) => {
                    e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*RqXFpiNGwdiKBWyLJc_E7g.png"
                  }}
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
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Types of Logistic Regression Visualization</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="/logistic-regression-types.png"
                  alt="Types of Logistic Regression"
                  className="max-w-full h-auto"
                  onError={(e) => {
                    e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*-3GtcUQkIGFpZBJWZ6-FDQ.jpeg"
                  }}
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
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-950 dark:to-cyan-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
          <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-3">Implementation in Python</h3>
          <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
            Let's implement logistic regression using Python and scikit-learn. We'll walk through the entire process
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
                <h4 className="font-medium text-lg">Complete Implementation Example</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onCopy(implementationCode, "implementation-code")}
                  className="text-xs"
                >
                  {copied === "implementation-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                <code>{implementationCode}</code>
              </pre>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Decision Boundary Visualization</h4>
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                <img
                  src="/logistic-regression-decision-boundary.png"
                  alt="Logistic Regression Decision Boundary"
                  className="max-w-full h-auto"
                  onError={(e) => {
                    e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*Vd9ZAqzz-lWnNbwUSQrH3g.png"
                  }}
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
              <h4 className="font-medium text-lg mb-3">ROC Curve and Confusion Matrix</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/roc-curve.png"
                    alt="ROC Curve"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*Uu-t4pOotRQFoyrfqEvIEg.png"
                    }}
                  />
                </div>
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/confusion-matrix.png"
                    alt="Confusion Matrix Heatmap"
                    className="max-w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*9LMgLQYHNl8K_jQtm7xzwA.png"
                    }}
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
              <Sigma className="h-5 w-5 text-primary" />
              Summary of Key Concepts
            </h4>
            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Logistic regression predicts probabilities for classification using the sigmoid function</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>It comes in three types: binary, multinomial, and ordinal</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Implementation in scikit-learn is straightforward with many customization options</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Evaluation metrics include accuracy, precision, recall, F1 score, and ROC-AUC</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Regularization helps prevent overfitting and improves generalization</span>
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
                  <strong>Support Vector Machines:</strong> Another powerful classification algorithm with a different
                  approach
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  2
                </Badge>
                <span>
                  <strong>Decision Trees and Random Forests:</strong> Tree-based methods that can capture non-linear
                  relationships
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  3
                </Badge>
                <span>
                  <strong>Neural Networks:</strong> Deep learning approaches for more complex classification tasks
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  4
                </Badge>
                <span>
                  <strong>Feature Engineering:</strong> Advanced techniques to improve model performance
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  5
                </Badge>
                <span>
                  <strong>Model Interpretability:</strong> Methods to better understand and explain your models
                </span>
              </li>
            </ul>
          </Card>
        </div>

        <Card className="p-5 bg-muted/30">
          <h4 className="font-medium text-lg mb-3">Strengths and Limitations</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="border rounded-lg p-3 border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-950">
              <h5 className="font-medium mb-2 text-sm text-green-700 dark:text-green-400">Strengths</h5>
              <ul className="text-xs space-y-1 list-disc pl-4 text-green-700 dark:text-green-400">
                <li>Simple and interpretable</li>
                <li>Efficient training process</li>
                <li>Works well with linearly separable data</li>
                <li>Provides probability estimates</li>
                <li>Less prone to overfitting than complex models</li>
              </ul>
            </div>
            <div className="border rounded-lg p-3 border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950">
              <h5 className="font-medium mb-2 text-sm text-red-700 dark:text-red-400">Limitations</h5>
              <ul className="text-xs space-y-1 list-disc pl-4 text-red-700 dark:text-red-400">
                <li>Assumes linear decision boundary</li>
                <li>May underperform with complex relationships</li>
                <li>Sensitive to outliers</li>
                <li>Requires feature scaling for optimal performance</li>
                <li>Struggles with highly imbalanced datasets</li>
              </ul>
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
        We're currently developing content for this section of the Logistic Regression tutorial. Check back soon!
      </p>
    </div>
  )
}

// Code examples as constants
const sigmoidCode = `import numpy as np
import matplotlib.pyplot as plt

# Generate data points for z
z = np.linspace(-10, 10, 100)

# Calculate sigmoid values
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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
plt.show()

# Example of using sigmoid for classification
print("Example predictions:")
print(f"For z = -3: σ(z) = {sigmoid(-3):.4f} → Likely class 0")
print(f"For z = 0: σ(z) = {sigmoid(0):.4f} → Decision boundary")
print(f"For z = 3: σ(z) = {sigmoid(3):.4f} → Likely class 1")`

const implementationCode = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Load the Iris dataset (using only two classes for binary classification)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
# Keep only two classes (0 and 1)
X = X[y != 2]
y = y[y != 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Print the model coefficients
print("Model Coefficients:")
for i, coef in enumerate(model.coef_[0]):
    print(f"Feature {i+1}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC: {auc:.4f}")

# Visualize the results
plt.figure(figsize=(12, 5))

# Plot confusion matrix
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names[:2],
            yticklabels=iris.target_names[:2])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plot ROC curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize decision boundary (using first two features for visualization)
plt.figure(figsize=(10, 8))

# Create a meshgrid to visualize decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Scale the meshgrid points
meshgrid_points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape), np.zeros(xx.ravel().shape)]
meshgrid_points_scaled = scaler.transform(meshgrid_points)

# Predict class for each point in the meshgrid
Z = model.predict(meshgrid_points_scaled)
Z = Z.reshape(xx.shape)

# Plot the decision boundary and training points
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.contour(xx, yy, Z, colors='k', linewidths=0.5)

# Plot the training points
for i, color in zip([0, 1], ['blue', 'red']):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                edgecolor='k', alpha=0.7)

plt.title('Logistic Regression Decision Boundary')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()`

const evaluationCode = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_curve, roc_auc_score)

# Load sample data (Breast Cancer dataset)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Print metrics
print("Model Evaluation Metrics:")
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
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Create visualizations
plt.figure(figsize=(15, 5))

# Plot confusion matrix
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plot ROC curve
plt.subplot(1, 2, 2)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Threshold analysis
thresholds = np.arange(0, 1.01, 0.1)
scores = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_threshold, zero_division=0)
    recall = recall_score(y_test, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_test, y_pred_threshold, zero_division=0)
    scores.append([precision, recall, f1])

scores_df = pd.DataFrame(scores, columns=['Precision', 'Recall', 'F1 Score'], index=thresholds)
print("\nMetrics at different thresholds:")
print(scores_df)

# Plot precision-recall tradeoff
plt.figure(figsize=(10, 6))
plt.plot(thresholds, scores_df['Precision'], 'b-', label='Precision')
plt.plot(thresholds, scores_df['Recall'], 'g-', label='Recall')
plt.plot(thresholds, scores_df['F1 Score'], 'r-', label='F1 Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Tradeoff')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()`

