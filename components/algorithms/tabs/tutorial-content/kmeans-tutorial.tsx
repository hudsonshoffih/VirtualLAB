"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { BookOpen, Code, BarChart, Lightbulb, CheckCircle, ArrowRight, ChevronLeft, ChevronRight, LineChart, GitBranch, Layers, Target, Copy, Check, PieChart, Image, Workflow, Repeat, Zap, AlertTriangle, Briefcase } from 'lucide-react'

interface kmeansTutorialProps {
  section: number
  onCopy: (text: string, id: string) => void
  copied: string | null
}

export function KMeansTutorial({ section, onCopy, copied }: kmeansTutorialProps) {
    const [copiedState, setCopiedState] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("explanation")

  // Define sections
  const sections = [
    { title: "Introduction to K-Means", icon: <Target className="h-4 w-4" /> },
    { title: "How K-Means Works", icon: <Workflow className="h-4 w-4" /> },
    { title: "Mathematical Concepts", icon: <LineChart className="h-4 w-4" /> },
    { title: "Choosing K Value", icon: <BarChart className="h-4 w-4" /> },
    { title: "Implementation", icon: <Code className="h-4 w-4" /> },
    { title: "Visualization", icon: <PieChart className="h-4 w-4" /> },
    { title: "Pros & Cons", icon: <Layers className="h-4 w-4" /> },
    { title: "Applications", icon: <Briefcase className="h-4 w-4" /> },
  ]

  // Code examples
  const elbowMethodCode = `import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Calculate WCSS for different values of K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal K', fontsize=16)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
plt.grid(True)
plt.xticks(range(1, 11))
plt.show()`

  const dataPreparationCode = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, 
                      cluster_std=0.6, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.7)
plt.title('Generated Dataset', fontsize=16)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.grid(True)
plt.show()`

  const kmeansImplementationCode = `from sklearn.cluster import KMeans

# Initialize and fit K-Means
kmeans = KMeans(n_clusters=4,      # Number of clusters
                init='k-means++',   # Initialization method
                max_iter=300,       # Maximum iterations
                n_init=10,          # Number of initializations
                random_state=42)    # For reproducibility

# Fit the model and predict cluster labels
y_kmeans = kmeans.fit_predict(X_scaled)

# Get cluster centers
centers = kmeans.cluster_centers_

# Get inertia (WCSS)
wcss = kmeans.inertia_
print(f"Within-Cluster Sum of Squares: {wcss:.2f}")`

  const visualizationCode = `import matplotlib.pyplot as plt
import numpy as np

# Create a colormap
colors = ['red', 'blue', 'green', 'cyan']

# Plot the clusters
plt.figure(figsize=(12, 8))

# Plot each cluster with a different color
for i in range(4):
    # Plot points in this cluster
    plt.scatter(X_scaled[y_kmeans == i, 0], 
                X_scaled[y_kmeans == i, 1], 
                s=100, c=colors[i], 
                label=f'Cluster {i+1}',
                alpha=0.7)

# Plot the centroids
plt.scatter(centers[:, 0], centers[:, 1], 
            s=300, c='yellow', 
            marker='*', 
            label='Centroids',
            edgecolor='black')

plt.title('K-Means Clustering Result', fontsize=16)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()`

  const silhouetteCode = `from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, y_kmeans)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Calculate silhouette values for each sample
sample_silhouette_values = silhouette_samples(X_scaled, y_kmeans)

# Plot silhouette analysis
plt.figure(figsize=(12, 8))
y_lower = 10

for i in range(4):
    # Get silhouette values for cluster i
    ith_cluster_values = sample_silhouette_values[y_kmeans == i]
    ith_cluster_values.sort()
    
    size_cluster_i = ith_cluster_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    # Fill the area with cluster color
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_values,
                      alpha=0.7, color=colors[i])
    
    # Label the silhouette plots
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i+1}')
    
    # Compute the new y_lower for next plot
    y_lower = y_upper + 10

plt.title('Silhouette Analysis', fontsize=16)
plt.xlabel('Silhouette Coefficient Values', fontsize=12)
plt.ylabel('Cluster Label', fontsize=12)

# Add vertical line for average silhouette score
plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.yticks([])  # Clear y-axis labels
plt.grid(True)
plt.show()`
    // Section 0: Introduction
    if (section === 0) {
      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-blue-100 dark:bg-blue-900 p-2 rounded-full">
                <Target className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300">
                What is K-Means Clustering?
              </h3>
            </div>
            <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
              K-Means Clustering is an unsupervised machine learning algorithm used for partitioning a dataset into a 
              set of distinct, non-overlapping groups called clusters. It works by minimizing the sum of distances 
              between data points and the centroids of their respective clusters. As one of the simplest and most 
              popular clustering algorithms, K-Means is widely used for data segmentation and pattern recognition.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center gap-2 mb-3">
                <Target className="h-5 w-5 text-blue-500" />
                <h4 className="font-medium text-lg">Key Concepts</h4>
              </div>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    1
                  </span>
                  <span>Clusters: Groups of similar data points</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    2
                  </span>
                  <span>Centroids: Center points of clusters</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    3
                  </span>
                  <span>Distance metrics: Measure of similarity</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    4
                  </span>
                  <span>Iterative refinement: Process of improving clusters</span>
                </li>
              </ul>
            </Card>

            <Card className="p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center gap-2 mb-3">
                <Lightbulb className="h-5 w-5 text-yellow-500" />
                <h4 className="font-medium text-lg">When to Use K-Means</h4>
              </div>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>When you need to discover groups in unlabeled data</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>For data segmentation and pattern recognition</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>When clusters are expected to be spherical</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                    •
                  </span>
                  <span>For large datasets where efficiency matters</span>
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
                { icon: <Workflow className="h-4 w-4" />, text: "K-Means Algorithm Steps" },
                { icon: <LineChart className="h-4 w-4" />, text: "Mathematical Foundations" },
                { icon: <BarChart className="h-4 w-4" />, text: "Selecting Optimal K" },
                { icon: <Code className="h-4 w-4" />, text: "Implementation with Python" },
                { icon: <PieChart className="h-4 w-4" />, text: "Cluster Visualization" },
                { icon: <Layers className="h-4 w-4" />, text: "Advantages & Limitations" },
                { icon: <Briefcase className="h-4 w-4" />, text: "Real-world Applications" },
                { icon: <Zap className="h-4 w-4" />, text: "Performance Optimization" },
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
                <span>Familiarity with NumPy and Matplotlib libraries</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-1 text-primary" />
                <span>Basic knowledge of data preprocessing techniques</span>
              </li>
              <li className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-1 text-primary" />
                <span>Understanding of basic statistical concepts</span>
              </li>
            </ul>
          </div>

          <div className="bg-muted/20 p-6 rounded-lg border mt-6">
            <h4 className="font-medium text-lg mb-4 flex items-center gap-2">
              <Target className="h-5 w-5 text-primary" />
              K-Means Clustering Visualization
            </h4>
            <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
              <img
                src="/placeholder.svg?height=400&width=800"
                alt="K-Means Clustering Visualization"
                className="max-w-full h-auto"
                onError={(e) => {
                  e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*KrcZK0xYgTa4qFrVr0fO2w.png"
                }}
              />
            </div>
            <p className="mt-4 text-sm text-muted-foreground">
              The visualization shows how K-Means clustering partitions data points into distinct groups. Each color 
              represents a different cluster, and the larger markers indicate the centroids (center points) of each 
              cluster. K-Means iteratively refines these clusters until convergence.
            </p>
          </div>
        </div>
      )
    }

    // Section 1: How K-Means Works
    if (section === 1) {
      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-blue-100 dark:bg-blue-900 p-2 rounded-full">
                <Workflow className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300">
                How K-Means Clustering Works
              </h3>
            </div>
            <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
              K-Means is an iterative algorithm that divides data into K clusters by minimizing the distance between 
              data points and their assigned cluster centers. The algorithm follows a simple procedure of initialization, 
              assignment, and update steps until convergence is reached.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Algorithm Steps
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <PieChart className="h-4 w-4 mr-2" />
                Visual Walkthrough
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Pseudocode
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <div className="bg-muted/30 p-5 rounded-lg border">
                <h4 className="font-medium text-lg mb-3">K-Means Algorithm Steps</h4>
                <ol className="space-y-6">
                  <li className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Initialization</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Choose K initial centroids (cluster centers). This can be done randomly or using methods like 
                        k-means++ that place initial centroids far apart from each other.
                      </p>
                      <div className="mt-2 bg-blue-50 dark:bg-blue-950 p-3 rounded-md">
                        <p className="text-xs text-blue-700 dark:text-blue-300">
                          <strong>Note:</strong> The initialization step is crucial as it can affect the final clustering 
                          result. Poor initialization may lead to suboptimal solutions.
                        </p>
                      </div>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Assignment Step</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Assign each data point to the nearest centroid based on a distance metric (typically Euclidean 
                        distance). This creates K clusters.
                      </p>
                      <div className="mt-2 p-3 bg-muted rounded-md">
                        <p className="text-xs font-mono">
                          For each data point x:
                          <br />
                          &nbsp;&nbsp;Calculate distance to each centroid
                          <br />
                          &nbsp;&nbsp;Assign x to the cluster with the closest centroid
                        </p>
                      </div>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Update Step</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Recalculate the centroids of each cluster by taking the mean of all data points assigned to 
                        that cluster.
                      </p>
                      <div className="mt-2 p-3 bg-muted rounded-md">
                        <p className="text-xs font-mono">
                          For each cluster k:
                          <br />
                          &nbsp;&nbsp;New centroid = mean of all points assigned to cluster k
                        </p>
                      </div>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Iteration and Convergence</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Repeat the assignment and update steps until convergence is reached. Convergence occurs when:
                      </p>
                      <ul className="mt-2 space-y-1 text-sm list-disc pl-5 text-muted-foreground">
                        <li>Centroids no longer move significantly</li>
                        <li>Data point assignments no longer change</li>
                        <li>Maximum number of iterations is reached</li>
                      </ul>
                    </div>
                  </li>
                </ol>
              </div>

              <Card className="p-5">
                <div className="flex items-center gap-2 mb-3">
                  <Lightbulb className="h-5 w-5 text-yellow-500" />
                  <h4 className="font-medium">Key Insights</h4>
                </div>
                <ul className="space-y-3 text-muted-foreground text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>
                      <span className="font-medium text-foreground">Guaranteed Convergence:</span> K-Means always 
                      converges to a local optimum, though not necessarily the global optimum.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>
                      <span className="font-medium text-foreground">Sensitivity to Initialization:</span> Different 
                      initial centroids can lead to different final clusters.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>
                      <span className="font-medium text-foreground">Efficiency:</span> The algorithm has a time 
                      complexity of O(n*K*d*i), where n is the number of points, K is the number of clusters, d is 
                      the number of dimensions, and i is the number of iterations.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <span>
                      <span className="font-medium text-foreground">Cluster Shape Assumption:</span> K-Means works 
                      best when clusters are spherical and of similar size.
                    </span>
                  </li>
                </ul>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Step 1: Initialization</h4>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=300&width=300"
                        alt="K-Means Initialization"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*KrcZK0xYgTa4qFrVr0fO2w.png"
                        }}
                      />
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">
                      Initial centroids (stars) are placed in the data space. Here, K=3 centroids are initialized.
                    </p>
                  </Card>

                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Step 2: Assignment</h4>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=300&width=300"
                        alt="K-Means Assignment"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*wfHRzXXkHyP-rUKYLbCZPg.png"
                        }}
                      />
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">
                      Each data point is assigned to the nearest centroid, forming initial clusters (shown in different colors).
                    </p>
                  </Card>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Step 3: Update</h4>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=300&width=300"
                        alt="K-Means Update"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*jQwP5y9VF9xMQYCNgY9jYA.png"
                        }}
                      />
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">
                      Centroids are recalculated as the mean of all points in each cluster. Notice how the centroids (stars) have moved to the center of their respective clusters.
                    </p>
                  </Card>

                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Step 4: Convergence</h4>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=300&width=300"
                        alt="K-Means Convergence"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*_RM6VlFKxXkQ-ymzglyTEQ.png"
                        }}
                      />
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">
                      After several iterations of assignment and update steps, the algorithm converges to stable clusters where centroids no longer move significantly.
                    </p>
                  </Card>
                </div>

                <div className="bg-muted/30 p-5 rounded-lg border">
                  <h4 className="font-medium text-lg mb-3">Animation of K-Means Process</h4>
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=800"
                      alt="K-Means Animation"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src = "https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif"
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    This animation shows the complete K-Means process from initialization to convergence. Watch how the centroids move and the cluster assignments change with each iteration until stability is reached.
                  </p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">K-Means Algorithm Pseudocode</h4>
                <div className="bg-muted p-4 rounded-lg overflow-x-auto text-sm font-mono">
                  <pre>
{`ALGORITHM K-Means(D, K)
  # D: Dataset with n data points
  # K: Number of clusters
  
  # Step 1: Initialization
  Initialize K centroids μ₁, μ₂, ..., μₖ randomly or using k-means++
  
  # Step 2-4: Iterative refinement
  repeat until convergence:
    
    # Step 2: Assignment step
    for each data point x in D:
      Assign x to the cluster C_i with the closest centroid μᵢ
      # Using Euclidean distance: argmin_i ||x - μᵢ||²
    
    # Step 3: Update step
    for each cluster C_i (i = 1 to K):
      Update centroid μᵢ = mean of all points assigned to C_i
    
    # Step 4: Check convergence
    if centroids don't change significantly or max iterations reached:
      break
  
  return clusters C₁, C₂, ..., Cₖ and centroids μ₁, μ₂, ..., μₖ`}
                  </pre>
                </div>
                <div className="mt-4 space-y-3">
                  <h5 className="font-medium">Implementation Notes:</h5>
                  <ul className="space-y-2 text-sm text-muted-foreground list-disc pl-5">
                    <li>
                      <span className="font-medium text-foreground">Initialization Methods:</span> Random initialization 
                      can lead to poor results. K-means++ initialization selects initial centroids that are far apart, 
                      which often leads to better clustering.
                    </li>
                    <li>
                      <span className="font-medium text-foreground">Empty Clusters:</span> If a cluster becomes empty 
                      during the algorithm, a common strategy is to reinitialize its centroid or assign the point 
                      farthest from its centroid.
                    </li>
                    <li>
                      <span className="font-medium text-foreground">Convergence Criteria:</span> Typically, the algorithm 
                      stops when centroids move less than a small threshold or after a maximum number of iterations.
                    </li>
                    <li>
                      <span className="font-medium text-foreground">Multiple Runs:</span> Due to sensitivity to 
                      initialization, it's common to run K-means multiple times with different initializations and 
                      select the result with the lowest within-cluster sum of squares.
                    </li>
                  </ul>
                </div>
              </Card>

              <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-lg border border-blue-100 dark:border-blue-900 mt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="h-5 w-5 text-blue-500" />
                  <h4 className="font-medium">K-means++ Initialization</h4>
                </div>
                <p className="text-sm text-blue-700 dark:text-blue-300 mb-3">
                  K-means++ is a smart initialization method that improves the standard K-means algorithm:
                </p>
                <ol className="space-y-2 text-sm text-blue-700 dark:text-blue-300 list-decimal pl-5">
                  <li>Choose the first centroid randomly from the data points</li>
                  <li>For each data point, compute the distance to the nearest existing centroid</li>
                  <li>Choose the next centroid from the data points with probability proportional to the squared distance</li>
                  <li>Repeat steps 2-3 until K centroids are chosen</li>
                </ol>
                <p className="text-sm text-blue-700 dark:text-blue-300 mt-3">
                  This approach ensures initial centroids are well-spread, leading to better final clustering results and 
                  faster convergence.
                </p>
              </div>
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
                <span>K-Means follows a simple iterative process: initialize, assign, update, repeat</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The algorithm always converges, but may find a local optimum rather than the global optimum</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Initialization is crucial - k-means++ initialization often leads to better results</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The algorithm minimizes the within-cluster sum of squares (inertia)</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 2: Mathematical Concepts
    if (section === 2) {
      function copyToClipboard(arg0: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-purple-100 dark:bg-purple-900 p-2 rounded-full">
                <LineChart className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300">
                Mathematical Concepts
              </h3>
            </div>
            <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
              Understanding the mathematical foundations of K-Means helps in grasping how the algorithm works and why 
              it makes certain decisions. The core mathematical concept behind K-Means is the minimization of the 
              Within-Cluster Sum of Squares (WCSS), also known as inertia.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Core Concepts
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Visual Explanation
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Mathematical Implementation
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Objective Function: WCSS</h4>
                <p className="text-muted-foreground mb-4">
                  The primary objective of K-Means is to minimize the Within-Cluster Sum of Squares (WCSS), which measures 
                  how compact the clusters are:
                </p>
                <div className="bg-muted/50 p-4 rounded-lg text-center mb-4">
                  <p className="font-medium">WCSS = ∑<sub>i=1</sub><sup>K</sup> ∑<sub>x∈C<sub>i</sub></sub> ||x - μ<sub>i</sub>||<sup>2</sup></p>
                </div>
                <p className="text-muted-foreground mb-2">Where:</p>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <span className="font-medium text-foreground">K:</span> Number of clusters
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-medium text-foreground">C<sub>i</sub>:</span> Set of points in cluster i
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-medium text-foreground">μ<sub>i</sub>:</span> Centroid (mean) of cluster i
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-medium text-foreground">||x - μ<sub>i</sub>||<sup>2</sup>:</span> Squared Euclidean distance between point x and centroid μ<sub>i</sub>
                  </li>
                </ul>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Distance Metrics</h4>
                  <p className="text-muted-foreground mb-3">
                    K-Means typically uses Euclidean distance to measure similarity between points, but other distance 
                    metrics can be used:
                  </p>
                  <div className="space-y-3">
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <h5 className="font-medium mb-1">Euclidean Distance</h5>
                      <p className="text-sm text-muted-foreground mb-2">
                        The straight-line distance between two points in Euclidean space:
                      </p>
                      <p className="text-center text-sm">
                        d(x, y) = √(∑<sub>i=1</sub><sup>n</sup> (x<sub>i</sub> - y<sub>i</sub>)<sup>2</sup>)
                      </p>
                    </div>
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <h5 className="font-medium mb-1">Manhattan Distance</h5>
                      <p className="text-sm text-muted-foreground mb-2">
                        The sum of absolute differences between coordinates:
                      </p>
                      <p className="text-center text-sm">
                        d(x, y) = ∑<sub>i=1</sub><sup>n</sup> |x<sub>i</sub> - y<sub>i</sub>|
                      </p>
                    </div>
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <h5 className="font-medium mb-1">Cosine Distance</h5>
                      <p className="text-sm text-muted-foreground mb-2">
                        Measures the cosine of the angle between two vectors:
                      </p>
                      <p className="text-center text-sm">
                        d(x, y) = 1 - (x·y) / (||x|| × ||y||)
                      </p>
                    </div>
                  </div>
                </Card>

                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Centroid Calculation</h4>
                  <p className="text-muted-foreground mb-3">
                    The centroid of a cluster is calculated as the mean of all points in that cluster:
                  </p>
                  <div className="bg-muted/50 p-4 rounded-lg text-center mb-4">
                    <p className="font-medium">μ<sub>i</sub> = (1/|C<sub>i</sub>|) ∑<sub>x∈C<sub>i</sub></sub> x</p>
                  </div>
                  <p className="text-muted-foreground mb-2">Where:</p>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <span className="font-medium text-foreground">μ<sub>i</sub>:</span> Centroid of cluster i
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-medium text-foreground">|C<sub>i</sub>|:</span> Number of points in cluster i
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-medium text-foreground">∑<sub>x∈C<sub>i</sub></sub> x:</span> Sum of all points in cluster i
                    </li>
                  </ul>
                  <div className="mt-3 bg-purple-50 dark:bg-purple-950 p-3 rounded-md">
                    <p className="text-xs text-purple-700 dark:text-purple-300">
                      <strong>Note:</strong> The centroid calculation is what makes K-Means a "means" algorithm - it uses 
                      the arithmetic mean to represent each cluster.
                    </p>
                  </div>
                </Card>
              </div>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Convergence Properties</h4>
                <p className="text-muted-foreground mb-3">
                  K-Means is guaranteed to converge to a local optimum due to the following mathematical properties:
                </p>
                <ul className="space-y-3 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <div>
                      <span className="font-medium text-foreground">Monotonic Decrease:</span> Each iteration of the 
                      algorithm decreases (or maintains) the WCSS value, never increasing it.
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <div>
                      <span className="font-medium text-foreground">Finite Configurations:</span> There are only a finite 
                      number of ways to assign n points to K clusters (K<sup>n</sup> possibilities).
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                    <div>
                      <span className="font-medium text-foreground">Lower Bound:</span> The WCSS is always non-negative 
                      and has a theoretical minimum of zero (when each point is its own centroid).
                    </div>
                  </li>
                </ul>
                <div className="mt-3 bg-muted/30 p-3 rounded-md">
                  <p className="text-sm">
                    <strong>Convergence Proof:</strong> Since the algorithm monotonically decreases a function that has a 
                    lower bound, and there are only a finite number of configurations, the algorithm must eventually 
                    converge to a local minimum.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <div className="space-y-6">
                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Visualizing WCSS</h4>
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=800"
                      alt="WCSS Visualization"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*_7gXP1P_JdHDFbrrI-S4XA.png"
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    This visualization shows how K-Means minimizes the Within-Cluster Sum of Squares. The circles represent 
                    the sum of squared distances from points to their respective centroids. The algorithm aims to minimize 
                    the total area of these circles.
                  </p>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Distance Metrics Comparison</h4>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=300&width=300"
                        alt="Distance Metrics"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*yGMk1GSKAJcXLnRxP7RJzg.png"
                        }}
                      />
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">
                      Different distance metrics create different "shapes" of neighborhoods. Euclidean distance (circle), 
                      Manhattan distance (diamond), and other metrics can lead to different clustering results.
                    </p>
                  </Card>

                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Centroid Calculation</h4>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=300&width=300"
                        alt="Centroid Calculation"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*G-tSrUIyi4-XLlbI_LP8Tg.png"
                        }}
                      />
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">
                      The centroid (star) is calculated as the mean of all points in the cluster. This minimizes the sum 
                      of squared distances from points to the centroid.
                    </p>
                  </Card>
                </div>

                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Convergence Visualization</h4>
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=800"
                      alt="Convergence Graph"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*KLJZzJC0r5uNGROr1AqjLQ.png"
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    This graph shows how the WCSS decreases with each iteration until convergence. Notice that the 
                    improvement becomes smaller with each iteration, eventually reaching a plateau where further 
                    iterations provide negligible improvement.
                  </p>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Mathematical Implementation in Python</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(`import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calculate_wcss(X, labels, centroids):
    """
    Calculate Within-Cluster Sum of Squares (WCSS)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
    centroids : array-like, shape (n_clusters, n_features)
        Cluster centroids
        
    Returns:
    --------
    wcss : float
        Within-Cluster Sum of Squares
    """
    n_clusters = centroids.shape[0]
    wcss = 0.0
    
    for i in range(n_clusters):
        # Get points in this cluster
        cluster_points = X[labels == i]
        
        if len(cluster_points) > 0:
            # Calculate squared distances to centroid
            centroid = centroids[i].reshape(1, -1)
            squared_distances = euclidean_distances(cluster_points, centroid, squared=True)
            
            # Sum the squared distances
            wcss += np.sum(squared_distances)
    
    return wcss

def calculate_centroids(X, labels, n_clusters):
    """
    Calculate centroids for each cluster
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    centroids : array-like, shape (n_clusters, n_features)
        Calculated centroids
    """
    n_features = X.shape[1]
    centroids = np.zeros((n_clusters, n_features))
    
    for i in range(n_clusters):
        # Get points in this cluster
        cluster_points = X[labels == i]
        
        if len(cluster_points) > 0:
            # Calculate mean of points (centroid)
            centroids[i] = np.mean(cluster_points, axis=0)
    
    return centroids

def kmeans_single_iteration(X, labels, centroids):
    """
    Perform a single iteration of K-Means
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Current cluster labels
    centroids : array-like, shape (n_clusters, n_features)
        Current centroids
        
    Returns:
    --------
    new_labels : array-like, shape (n_samples,)
        Updated cluster labels
    new_centroids : array-like, shape (n_clusters, n_features)
        Updated centroids
    wcss : float
        Within-Cluster Sum of Squares after update
    """
    n_clusters = centroids.shape[0]
    
    # Assignment step: assign each point to nearest centroid
    distances = euclidean_distances(X, centroids, squared=True)
    new_labels = np.argmin(distances, axis=1)
    
    # Update step: recalculate centroids
    new_centroids = calculate_centroids(X, new_labels, n_clusters)
    
    # Calculate WCSS
    wcss = calculate_wcss(X, new_labels, new_centroids)
    
    return new_labels, new_centroids, wcss`, "math-implementation")}
                    className="text-xs"
                  >
                    {copiedState === "math-implementation" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{`import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calculate_wcss(X, labels, centroids):
    """
    Calculate Within-Cluster Sum of Squares (WCSS)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
    centroids : array-like, shape (n_clusters, n_features)
        Cluster centroids
        
    Returns:
    --------
    wcss : float
        Within-Cluster Sum of Squares
    """
    n_clusters = centroids.shape[0]
    wcss = 0.0
    
    for i in range(n_clusters):
        # Get points in this cluster
        cluster_points = X[labels == i]
        
        if len(cluster_points) > 0:
            # Calculate squared distances to centroid
            centroid = centroids[i].reshape(1, -1)
            squared_distances = euclidean_distances(cluster_points, centroid, squared=True)
            
            # Sum the squared distances
            wcss += np.sum(squared_distances)
    
    return wcss

def calculate_centroids(X, labels, n_clusters):
    """
    Calculate centroids for each cluster
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    centroids : array-like, shape (n_clusters, n_features)
        Calculated centroids
    """
    n_features = X.shape[1]
    centroids = np.zeros((n_clusters, n_features))
    
    for i in range(n_clusters):
        # Get points in this cluster
        cluster_points = X[labels == i]
        
        if len(cluster_points) > 0:
            # Calculate mean of points (centroid)
            centroids[i] = np.mean(cluster_points, axis=0)
    
    return centroids

def kmeans_single_iteration(X, labels, centroids):
    """
    Perform a single iteration of K-Means
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    labels : array-like, shape (n_samples,)
        Current cluster labels
    centroids : array-like, shape (n_clusters, n_features)
        Current centroids
        
    Returns:
    --------
    new_labels : array-like, shape (n_samples,)
        Updated cluster labels
    new_centroids : array-like, shape (n_clusters, n_features)
        Updated centroids
    wcss : float
        Within-Cluster Sum of Squares after update
    """
    n_clusters = centroids.shape[0]
    
    # Assignment step: assign each point to nearest centroid
    distances = euclidean_distances(X, centroids, squared=True)
    new_labels = np.argmin(distances, axis=1)
    
    # Update step: recalculate centroids
    new_centroids = calculate_centroids(X, new_labels, n_clusters)
    
    # Calculate WCSS
    wcss = calculate_wcss(X, new_labels, new_centroids)
    
    return new_labels, new_centroids, wcss`}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code demonstrates the mathematical implementation of K-Means, including functions to calculate 
                    the Within-Cluster Sum of Squares (WCSS), compute centroids, and perform a single iteration of the 
                    K-Means algorithm.
                  </p>
                </div>
              </Card>

              <div className="bg-purple-50 dark:bg-purple-950 p-4 rounded-lg border border-purple-100 dark:border-purple-900 mt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="h-5 w-5 text-purple-500" />
                  <h4 className="font-medium">Time Complexity Analysis</h4>
                </div>
                <p className="text-sm text-purple-700 dark:text-purple-300 mb-3">
                  The time complexity of K-Means can be broken down as follows:
                </p>
                <ul className="space-y-2 text-sm text-purple-700 dark:text-purple-300">
                  <li className="flex items-start gap-2">
                    <span className="font-medium">O(n * K * d * i)</span> where:
                    <ul className="ml-4 space-y-1">
                      <li>n = number of data points</li>
                      <li>K = number of clusters</li>
                      <li>d = number of dimensions</li>
                      <li>i = number of iterations</li>
                    </ul>
                  </li>
                  <li>The assignment step takes O(n * K * d) time to compute distances between all points and centroids</li>
                  <li>The update step takes O(n * d) time to recalculate centroids</li>
                  <li>This is repeated for i iterations until convergence</li>
                </ul>
                <p className="text-sm text-purple-700 dark:text-purple-300 mt-3">
                  For large datasets, optimizations like KD-Trees or Ball Trees can reduce the complexity to approximately 
                  O(n * log(K) * d * i) in many practical cases.
                </p>
              </div>
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
                <span>K-Means minimizes the Within-Cluster Sum of Squares (WCSS) to create compact clusters</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The centroid of a cluster is the mean of all points in that cluster</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Euclidean distance is commonly used, but other distance metrics can be applied</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>K-Means is guaranteed to converge to a local optimum, but not necessarily the global optimum</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 3: Choosing K Value
    if (section === 3) {
      function copyToClipboard(silhouetteCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950 dark:to-orange-950 p-6 rounded-lg border border-amber-100 dark:border-amber-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-amber-100 dark:bg-amber-900 p-2 rounded-full">
                <BarChart className="h-6 w-6 text-amber-600 dark:text-amber-400" />
              </div>
              <h3 className="text-xl font-semibold text-amber-800 dark:text-amber-300">
                Choosing the Optimal K Value
              </h3>
            </div>
            <p className="text-amber-700 dark:text-amber-300 leading-relaxed">
              One of the main challenges in K-Means clustering is determining the optimal number of clusters (K). 
              Since K-Means requires specifying K in advance, choosing an appropriate value is crucial for meaningful 
              results. Several methods can help identify the optimal K value for your dataset.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Methods
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Implementation
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Visualization
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <BarChart className="h-5 w-5 text-amber-500" />
                    <h4 className="font-medium text-lg">Elbow Method</h4>
                  </div>
                  <p className="text-muted-foreground mb-3">
                    The Elbow Method plots the Within-Cluster Sum of Squares (WCSS) against different K values and 
                    identifies the "elbow point" where the rate of decrease sharply changes.
                  </p>
                  <div className="bg-muted/30 p-3 rounded-lg">
                    <h5 className="font-medium mb-1">How it works:</h5>
                    <ol className="text-sm text-muted-foreground space-y-1 list-decimal pl-5">
                      <li>Run K-Means with increasing values of K (e.g., 1 to 10)</li>
                      <li>For each K, calculate and plot the WCSS</li>
                      <li>Look for the point where the curve bends (the "elbow")</li>
                      <li>This point represents a good trade-off between cluster quality and complexity</li>
                    </ol>
                  </div>
                  <div className="mt-3 bg-amber-50 dark:bg-amber-950 p-3 rounded-md">
                    <p className="text-xs text-amber-700 dark:text-amber-300">
                      <strong>Intuition:</strong> As K increases, WCSS always decreases (in the extreme case, when K equals 
                      the number of data points, WCSS becomes zero). The elbow point indicates where adding more clusters 
                      doesn't significantly reduce WCSS.
                    </p>
                  </div>
                </Card>

                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <LineChart className="h-5 w-5 text-blue-500" />
                    <h4 className="font-medium text-lg">Silhouette Method</h4>
                  </div>
                  <p className="text-muted-foreground mb-3">
                    The Silhouette Method measures how similar an object is to its own cluster compared to other clusters, 
                    providing a way to assess both cohesion and separation.
                  </p>
                  <div className="bg-muted/30 p-3 rounded-lg">
                    <h5 className="font-medium mb-1">How it works:</h5>
                    <ol className="text-sm text-muted-foreground space-y-1 list-decimal pl-5">
                      <li>For each data point i, calculate its silhouette coefficient s(i)</li>
                      <li>s(i) = (b(i) - a(i)) / max(a(i), b(i))</li>
                      <li>Where a(i) is the average distance to points in the same cluster</li>
                      <li>And b(i) is the average distance to points in the nearest different cluster</li>
                      <li>Average s(i) across all points to get the silhouette score</li>
                      <li>Choose K that maximizes the silhouette score</li>
                    </ol>
                  </div>
                  <div className="mt-3 bg-blue-50 dark:bg-blue-950 p-3 rounded-md">
                    <p className="text-xs text-blue-700 dark:text-blue-300">
                      <strong>Interpretation:</strong> Silhouette scores range from -1 to 1. Values close to 1 indicate 
                      well-separated clusters, values near 0 indicate overlapping clusters, and negative values suggest 
                      points may be assigned to the wrong cluster.
                    </p>
                  </div>
                </Card>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <GitBranch className="h-5 w-5 text-green-500" />
                    <h4 className="font-medium text-lg">Gap Statistic</h4>
                  </div>
                  <p className="text-muted-foreground mb-3">
                    The Gap Statistic compares the within-cluster dispersion to that expected under a null reference 
                    distribution (i.e., no clustering).
                  </p>
                  <div className="bg-muted/30 p-3 rounded-lg">
                    <h5 className="font-medium mb-1">How it works:</h5>
                    <ol className="text-sm text-muted-foreground space-y-1 list-decimal pl-5">
                      <li>For each K, compute the within-cluster dispersion W<sub>k</sub></li>
                      <li>Generate B reference datasets with no clustering structure</li>
                      <li>Compute the dispersion W<sub>kb</sub> for each reference dataset</li>
                      <li>Calculate the gap statistic: Gap(k) = (1/B)∑log(W<sub>kb</sub>) - log(W<sub>k</sub>)</li>
                      <li>Choose K that maximizes the gap statistic</li>
                    </ol>
                  </div>
                  <div className="mt-3 bg-green-50 dark:bg-green-950 p-3 rounded-md">
                    <p className="text-xs text-green-700 dark:text-green-300">
                      <strong>Advantage:</strong> The Gap Statistic can identify when there is only one cluster (K=1) in 
                      the data, which other methods may struggle with. It's more statistically rigorous but 
                      computationally intensive.
                    </p>
                  </div>
                </Card>

                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Lightbulb className="h-5 w-5 text-purple-500" />
                    <h4 className="font-medium text-lg">Other Methods</h4>
                  </div>
                  <div className="space-y-3">
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <h5 className="font-medium mb-1">Calinski-Harabasz Index</h5>
                      <p className="text-sm text-muted-foreground">
                        Also known as the Variance Ratio Criterion, it measures the ratio of between-cluster dispersion to 
                        within-cluster dispersion. Higher values indicate better clustering.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <h5 className="font-medium mb-1">Davies-Bouldin Index</h5>
                      <p className="text-sm text-muted-foreground">
                        Measures the average similarity between clusters, where similarity is the ratio of within-cluster 
                        distances to between-cluster distances. Lower values indicate better clustering.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <h5 className="font-medium mb-1">Domain Knowledge</h5>
                      <p className="text-sm text-muted-foreground">
                        Sometimes the best approach is to use domain expertise to determine a reasonable number of 
                        clusters based on the specific application context.
                      </p>
                    </div>
                  </div>
                </Card>
              </div>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Comparison of Methods</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full border-collapse">
                    <thead>
                      <tr className="bg-muted/50">
                        <th className="border px-4 py-2 text-left">Method</th>
                        <th className="border px-4 py-2 text-left">Pros</th>
                        <th className="border px-4 py-2 text-left">Cons</th>
                        <th className="border px-4 py-2 text-left">Best For</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Elbow Method</td>
                        <td className="border px-4 py-2 text-sm">
                          <ul className="list-disc pl-4 space-y-1">
                            <li>Simple and intuitive</li>
                            <li>Easy to implement</li>
                            <li>Computationally efficient</li>
                          </ul>
                        </td>
                        <td className="border px-4 py-2 text-sm">
                          <ul className="list-disc pl-4 space-y-1">
                            <li>Subjective interpretation</li>
                            <li>Elbow may not be clear</li>
                            <li>Only considers WCSS</li>
                          </ul>
                        </td>
                        <td className="border px-4 py-2 text-sm">
                          <ul className="list-disc pl-4 space-y-1">
                            <li>Quick exploratory analysis</li>
                            <li>Well-separated clusters</li>
                            <li>Large datasets</li>
                          </ul>
                        </td>
                      </tr>
                      <tr className="bg-muted/20">
                        <td className="border px-4 py-2 font-medium">Silhouette Method</td>
                        <td className="border px-4 py-2 text-sm">
                          <ul className="list-disc pl-4 space-y-1">
                            <li>Considers both cohesion and separation</li>
                            <li>Provides score for each point</li>
                            <li>Interpretable scale (-1 to 1)</li>
                          </ul>
                        </td>
                        <td className="border px-4 py-2 text-sm">
                          <ul className="list-disc pl-4 space-y-1">
                            <li>Computationally intensive</li>
                            <li>Sensitive to cluster shape</li>
                            <li>May favor spherical clusters</li>
                          </ul>
                        </td>
                        <td className="border px-4 py-2 text-sm">
                          <ul className="list-disc pl-4 space-y-1">
                            <li>Validating cluster quality</li>
                            <li>Comparing different clustering results</li>
                            <li>Medium-sized datasets</li>
                          </ul>
                        </td>
                      </tr>
                      <tr>
                        <td className="border px-4 py-2 font-medium">Gap Statistic</td>
                        <td className="border px-4 py-2 text-sm">
                          <ul className="list-disc pl-4 space-y-1">
                            <li>Statistically rigorous</li>
                            <li>Can identify K=1</li>
                            <li>Compares to null reference</li>
                          </ul>
                        </td>
                        <td className="border px-4 py-2 text-sm">
                          <ul className="list-disc pl-4 space-y-1">
                            <li>Very computationally intensive</li>
                            <li>Complex implementation</li>
                            <li>Requires multiple simulations</li>
                          </ul>
                        </td>
                        <td className="border px-4 py-2 text-sm">
                          <ul className="list-disc pl-4 space-y-1">
                            <li>Rigorous statistical analysis</li>
                            <li>When other methods are inconclusive</li>
                            <li>Small to medium datasets</li>
                          </ul>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Elbow Method Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(elbowMethodCode, "elbow-method")}
                    className="text-xs"
                  >
                    {copiedState === "elbow-method" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{elbowMethodCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code calculates the Within-Cluster Sum of Squares (WCSS) for different values of K and plots 
                    the results to identify the elbow point. The elbow point indicates a good trade-off between cluster 
                    quality and complexity.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Silhouette Analysis</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(silhouetteCode, "silhouette-code")}
                    className="text-xs"
                  >
                    {copiedState === "silhouette-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{silhouetteCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code calculates and visualizes the silhouette score for a K-Means clustering result. The 
                    silhouette plot shows how well each point fits within its assigned cluster compared to other clusters.
                  </p>
                </div>
              </Card>

              <div className="bg-amber-50 dark:bg-amber-950 p-4 rounded-lg border border-amber-100 dark:border-amber-900 mt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="h-5 w-5 text-amber-500" />
                  <h4 className="font-medium">Comprehensive Approach</h4>
                </div>
                <p className="text-sm text-amber-700 dark:text-amber-300 mb-3">
                  For a more robust analysis, it's recommended to use multiple methods to determine the optimal K value:
                </p>
                <pre className="bg-amber-100/50 dark:bg-amber-900/50 p-3 rounded-md text-xs overflow-x-auto">
                  <code>{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Generate or load your data
# X = ...

# Range of K values to try
k_range = range(2, 11)

# Store metrics
wcss = []
silhouette_scores = []
ch_scores = []
db_scores = []

# Calculate metrics for each K
for k in k_range:
    # Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    # Calculate WCSS (inertia)
    wcss.append(kmeans.inertia_)
    
    # Calculate Silhouette Score
    silhouette_scores.append(silhouette_score(X, labels))
    
    # Calculate Calinski-Harabasz Index
    ch_scores.append(calinski_harabasz_score(X, labels))
    
    # Calculate Davies-Bouldin Index
    db_scores.append(davies_bouldin_score(X, labels))

# Plot all metrics
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# WCSS (Elbow Method)
axs[0, 0].plot(k_range, wcss, marker='o')
axs[0, 0].set_title('Elbow Method')
axs[0, 0].set_xlabel('Number of Clusters')
axs[0, 0].set_ylabel('WCSS')

# Silhouette Score
axs[0, 1].plot(k_range, silhouette_scores, marker='o')
axs[0, 1].set_title('Silhouette Method')
axs[0, 1].set_xlabel('Number of Clusters')
axs[0, 1].set_ylabel('Silhouette Score')

# Calinski-Harabasz Index
axs[1, 0].plot(k_range, ch_scores, marker='o')
axs[1, 0].set_title('Calinski-Harabasz Index')
axs[1, 0].set_xlabel('Number of Clusters')
axs[1, 0].set_ylabel('CH Score')

# Davies-Bouldin Index
axs[1, 1].plot(k_range, db_scores, marker='o')
axs[1, 1].set_title('Davies-Bouldin Index')
axs[1, 1].set_xlabel('Number of Clusters')
axs[1, 1].set_ylabel('DB Score')

plt.tight_layout()
plt.show()

# Print optimal K for each method
print(f"Optimal K according to Elbow Method: Look for the elbow point in the plot")
print(f"Optimal K according to Silhouette Method: {k_range[np.argmax(silhouette_scores)]}")
print(f"Optimal K according to Calinski-Harabasz Index: {k_range[np.argmax(ch_scores)]}")
print(f"Optimal K according to Davies-Bouldin Index: {k_range[np.argmin(db_scores)]}")`}</code>
                </pre>
                <p className="text-sm text-amber-700 dark:text-amber-300 mt-3">
                  By comparing the results from multiple methods, you can make a more informed decision about the optimal 
                  number of clusters for your specific dataset.
                </p>
              </div>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <div className="space-y-6">
                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Elbow Method Visualization</h4>
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=800"
                      alt="Elbow Method"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*b0_6WxjUPGzF1LXDl1Qiwg.png"
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    The Elbow Method plot shows WCSS decreasing as K increases. The "elbow" point (around K=4 in this 
                    example) indicates where adding more clusters doesn't significantly reduce WCSS. This suggests that 
                    4 is a good choice for the number of clusters.
                  </p>
                </Card>

                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Silhouette Analysis</h4>
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=800"
                      alt="Silhouette Analysis"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_silhouette_analysis_001.png"
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    Silhouette plots show how well each point fits within its assigned cluster. The average silhouette 
                    width (dashed red line) indicates overall clustering quality. Higher values and more uniform 
                    silhouette widths suggest better clustering. In this example, K=4 provides the best silhouette score.
                  </p>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Comparing Different K Values</h4>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=400&width=400"
                        alt="K Value Comparison"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*tMPUZNBGWEJjRtBUBiXy0g.png"
                        }}
                      />
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">
                      This visualization shows how the clustering results change with different K values. As K increases, 
                      clusters become smaller and more numerous. The optimal K balances between underfitting (too few 
                      clusters) and overfitting (too many clusters).
                    </p>
                  </Card>

                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Multiple Metrics Comparison</h4>
                    <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                      <img
                        src="/placeholder.svg?height=400&width=400"
                        alt="Multiple Metrics"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          e.currentTarget.src = "https://miro.medium.com/v2/resize:fit:1400/1*ET8kCcPpr893vNZFs8j4xg.png"
                        }}
                      />
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">
                      Comparing multiple evaluation metrics provides a more comprehensive view. This plot shows different 
                      metrics (Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index) for various K values. 
                      When multiple metrics agree on a particular K value, it strengthens the case for that choice.
                    </p>
                  </Card>
                </div>

                <div className="bg-muted/30 p-5 rounded-lg border">
                  <h4 className="font-medium text-lg mb-3">Practical Tips for Choosing K</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-amber-50 dark:bg-amber-950 p-4 rounded-lg border border-amber-100 dark:border-amber-900">
                      <h5 className="font-medium mb-2">Do's</h5>
                      <ul className="space-y-2 text-sm text-amber-700 dark:text-amber-300">
                        <li className="flex items-start gap-2">
                          <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                          <span>Use multiple methods to validate your choice of K</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                          <span>Consider domain knowledge when interpreting results</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                          <span>Visualize the clusters to assess their meaningfulness</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <CheckCircle className="h-4 w-4 text-green-500 mt-0.5" />
                          <span>Run K-Means multiple times with different initializations</span>
                        </li>
                      </ul>
                    </div>
                    <div className="bg-red-50 dark:bg-red-950 p-4 rounded-lg border border-red-100 dark:border-red-900">
                      <h5 className="font-medium mb-2">Don'ts</h5>
                      <ul className="space-y-2 text-sm text-red-700 dark:text-red-300">
                        <li className="flex items-start gap-2">
                          <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5" />
                          <span>Rely solely on one method for determining K</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5" />
                          <span>Choose a very high K value just to minimize WCSS</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5" />
                          <span>Ignore domain knowledge in favor of purely statistical methods</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5" />
                          <span>Assume that the optimal K is the same for all subsets of your data</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
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
                <span>Choosing the right K value is crucial for meaningful clustering results</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The Elbow Method identifies where adding more clusters provides diminishing returns</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Silhouette Analysis measures how well-separated clusters are</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Using multiple methods and domain knowledge leads to more robust K selection</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 4: Implementation
    if (section === 4) {
      function copyToClipboard(dataPreparationCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-950 dark:to-teal-950 p-6 rounded-lg border border-green-100 dark:border-green-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-green-100 dark:bg-green-900 p-2 rounded-full">
                <Code className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
              <h3 className="text-xl font-semibold text-green-800 dark:text-green-300">
                Implementation with Python
              </h3>
            </div>
            <p className="text-green-700 dark:text-green-300 leading-relaxed">
              Implementing K-Means clustering in Python is straightforward using libraries like scikit-learn. This 
              section covers the complete implementation process, from data preparation to model evaluation, with 
              practical examples and best practices.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Implementation Steps
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Code Examples
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Results
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Implementation Overview</h4>
                <ol className="space-y-4">
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Data Preparation</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Load, clean, and preprocess the data, including handling missing values, encoding categorical 
                        features, and scaling numerical features.
                      </p>
                      <div className="mt-2 bg-green-50 dark:bg-green-950 p-3 rounded-md">
                        <p className="text-xs text-green-700 dark:text-green-300">
                          <strong>Important:</strong> Feature scaling is crucial for K-Means since the algorithm uses 
                          distance metrics. Standardization (z-score normalization) or Min-Max scaling is commonly used.
                        </p>
                      </div>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Determining the Optimal K</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Use methods like the Elbow Method or Silhouette Analysis to identify the optimal number of clusters.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Model Training</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Initialize and train the K-Means model with the chosen K value and other parameters.
                      </p>
                      <div className="mt-2 p-3 bg-muted rounded-md">
                        <p className="text-xs font-mono">
                          from sklearn.cluster import KMeans
                          <br />
                          <br />
                          kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
                          <br />
                          kmeans.fit(X)
                        </p>
                      </div>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Cluster Assignment</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Assign each data point to a cluster and analyze the cluster centroids.
                      </p>
                      <div className="mt-2 p-3 bg-muted rounded-md">
                        <p className="text-xs font-mono">
                          # Get cluster labels for each data point
                          <br />
                          labels = kmeans.labels_
                          <br />
                          <br />
                          # Get cluster centers
                          <br />
                          centers = kmeans.cluster_centers_
                        </p>
                      </div>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      5
                    </div>
                    <div>
                      <h5 className="font-medium">Evaluation and Visualization</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Evaluate the clustering results using metrics like inertia, silhouette score, and visualize the 
                        clusters.
                      </p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      6
                    </div>
                    <div>
                      <h5 className="font-medium">Interpretation and Application</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Interpret the clustering results in the context of the problem domain and apply insights for 
                        decision-making.
                      </p>
                    </div>
                  </li>
                </ol>
              </Card>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Key Parameters in scikit-learn's KMeans</h4>
                <div className="space-y-3">
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      n_clusters
                    </Badge>
                    <div>
                      <p className="text-sm">Number of clusters (K) to form and centroids to generate</p>
                      <p className="text-xs text-muted-foreground">Default: 8</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      init
                    </Badge>
                    <div>
                      <p className="text-sm">Method for initialization of centroids</p>
                      <p className="text-xs text-muted-foreground">Options: 'k-means++' (default), 'random', or an ndarray</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      n_init
                    </Badge>
                    <div>
                      <p className="text-sm">Number of times the algorithm will run with different centroid seeds</p>
                      <p className="text-xs text-muted-foreground">Default: 10</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      max_iter
                    </Badge>
                    <div>
                      <p className="text-sm">Maximum number of iterations for a single run</p>
                      <p className="text-xs text-muted-foreground">Default: 300</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      tol
                    </Badge>
                    <div>
                      <p className="text-sm">Tolerance for declaring convergence</p>
                      <p className="text-xs text-muted-foreground">Default: 1e-4</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="mt-1">
                      random_state
                    </Badge>
                    <div>
                      <p className="text-sm">Seed for random number generation</p>
                      <p className="text-xs text-muted-foreground">For reproducibility</p>
                    </div>
                  </div>
                </div>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <h4 className="font-medium">Best Practices</h4>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Always scale your features before applying K-Means</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Run the algorithm multiple times (n_init) to avoid local minima</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Use k-means++ initialization for better starting centroids</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Set random_state for reproducibility</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-primary mt-0.5" />
                      <span>Consider dimensionality reduction for high-dimensional data</span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle className="h-5 w-5 text-amber-500" />
                    <h4 className="font-medium">Common Pitfalls</h4>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span>Not scaling features, leading to biased clustering</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span>Using K-Means for non-spherical or uneven clusters</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span>Ignoring outliers, which can significantly affect centroids</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span>Using too many features without dimensionality reduction</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <ArrowRight className="h-4 w-4 text-amber-500 mt-0.5" />
                      <span>Misinterpreting cluster assignments as ground truth</span>
                    </li>
                  </ul>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Data Preparation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(dataPreparationCode, "data-preparation")}
                    className="text-xs"
                  >
                    {copiedState === "data-preparation" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{dataPreparationCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code generates synthetic data with 4 clusters using scikit-learn's make_blobs function. It then 
                    standardizes the features using StandardScaler, which is an important preprocessing step for K-Means 
                    clustering.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">K-Means Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(kmeansImplementationCode, "kmeans-implementation")}
                    className="text-xs"
                  >
                    {copiedState === "kmeans-implementation" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{kmeansImplementationCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code initializes and fits a K-Means model with 4 clusters using the k-means++ initialization 
                    method. It then retrieves the cluster labels and centroids, and calculates the Within-Cluster Sum of 
                    Squares (WCSS).
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Visualization</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(visualizationCode, "visualization-code")}
                    className="text-xs"
                  >
                    {copiedState === "visualization-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{visualizationCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code creates a scatter plot to visualize the clustering results. Each cluster is represented by a 
                    different color, and the centroids are marked with yellow stars. This visualization helps in 
                    understanding how the data is partitioned into clusters.
                  </p>
                </div>
              </Card>

              <div className="bg-green-50 dark:bg-green-950 p-4 rounded-lg border border-green-100 dark:border-green-900 mt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="h-5 w-5 text-green-500" />
                  <h4 className="font-medium">Complete Example</h4>
                </div>
                <p className="text-sm text-green-700 dark:text-green-300 mb-3">
                  Here's a complete example that combines all the steps:
                </p>
                <pre className="bg-green-100/50 dark:bg-green-900/50 p-3 rounded-md text-xs overflow-x-auto">
                  <code>{`import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, 
                      cluster_std=0.6, random_state=42)

# 2. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Determine optimal K using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal K', fontsize=16)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
plt.grid(True)
plt.xticks(range(1, 11))
plt.show()

# 4. Train K-Means with the optimal K (in this case, K=4)
kmeans = KMeans(n_clusters=4, init='k-means++', 
               max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Get cluster centers
centers = kmeans.cluster_centers_

# 5. Evaluate the clustering
silhouette_avg = silhouette_score(X_scaled, y_kmeans)
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Within-Cluster Sum of Squares: {kmeans.inertia_:.2f}")

# 6. Visualize the clusters
colors = ['red', 'blue', 'green', 'cyan']

plt.figure(figsize=(12, 8))

# Plot each cluster with a different color
for i in range(4):
    plt.scatter(X_scaled[y_kmeans == i, 0], 
                X_scaled[y_kmeans == i, 1], 
                s=100, c=colors[i], 
                label=f'Cluster {i+1}',
                alpha=0.7)

# Plot the centroids
plt.scatter(centers[:, 0], centers[:, 1], 
            s=300, c='yellow', 
            marker='*', 
            label='Centroids',
            edgecolor='black')

plt.title('K-Means Clustering Result', fontsize=16)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

# 7. Analyze cluster characteristics
cluster_df = pd.DataFrame(X_scaled, columns=['Feature 1', 'Feature 2'])
cluster_df['Cluster'] = y_kmeans

# Calculate cluster statistics
cluster_stats = cluster_df.groupby('Cluster').agg(['mean', 'std', 'count'])
print("\nCluster Statistics:")
print(cluster_stats)`}</code>
                </pre>
                <p className="text-sm text-green-700 dark:text-green-300 mt-3">
                  This comprehensive example covers all the steps from data generation to cluster analysis, including 
                  determining the optimal K value, training the model, evaluating the results, and visualizing the clusters.
                </p>
              </div>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <div className="space-y-6">
                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Clustering Results</h4>
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=800"
                      alt="K-Means Clustering Results"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_digits_001.png"
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    This visualization shows the clustering results of K-Means on a 2D dataset. Each color represents a 
                    different cluster, and the yellow stars mark the cluster centroids. The clear separation between 
                    clusters indicates that K-Means has successfully identified the natural groupings in the data.
                  </p>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Cluster Characteristics</h4>
                    <div className="overflow-x-auto">
                      <table className="min-w-full border-collapse">
                        <thead>
                          <tr className="bg-muted/50">
                            <th className="border px-4 py-2 text-left">Cluster</th>
                            <th className="border px-4 py-2 text-left">Size</th>
                            <th className="border px-4 py-2 text-left">Feature 1 (Mean)</th>
                            <th className="border px-4 py-2 text-left">Feature 2 (Mean)</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td className="border px-4 py-2 font-medium">Cluster 0</td>
                            <td className="border px-4 py-2">75</td>
                            <td className="border px-4 py-2">-0.87</td>
                            <td className="border px-4 py-2">1.23</td>
                          </tr>
                          <tr className="bg-muted/20">
                            <td className="border px-4 py-2 font-medium">Cluster 1</td>
                            <td className="border px-4 py-2">78</td>
                            <td className="border px-4 py-2">1.45</td>
                            <td className="border px-4 py-2">0.92</td>
                          </tr>
                          <tr>
                            <td className="border px-4 py-2 font-medium">Cluster 2</td>
                            <td className="border px-4 py-2">72</td>
                            <td className="border px-4 py-2">-0.65</td>
                            <td className="border px-4 py-2">-1.08</td>
                          </tr>
                          <tr className="bg-muted/20">
                            <td className="border px-4 py-2 font-medium">Cluster 3</td>
                            <td className="border px-4 py-2">75</td>
                            <td className="border px-4 py-2">0.98</td>
                            <td className="border px-4 py-2">-1.17</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <p className="mt-3 text-sm text-muted-foreground">
                      This table shows the characteristics of each cluster, including the number of data points and the 
                      mean values of the features. Analyzing these statistics helps in understanding the differences 
                      between clusters.
                    </p>
                  </Card>

                  <Card className="p-5">
                    <h4 className="font-medium text-lg mb-3">Evaluation Metrics</h4>
                    <div className="space-y-4">
                      <div className="bg-muted/30 p-3 rounded-lg">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-medium">Silhouette Score</span>
                          <Badge variant="outline">0.723</Badge>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Measures how similar points are to their own cluster compared to other clusters. Higher values 
                          (closer to 1) indicate well-separated clusters.
                        </p>
                      </div>
                      <div className="bg-muted/30 p-3 rounded-lg">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-medium">Inertia (WCSS)</span>
                          <Badge variant="outline">142.36</Badge>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Sum of squared distances of samples to their closest cluster center. Lower values indicate more 
                          compact clusters.
                        </p>
                      </div>
                      <div className="bg-muted/30 p-3 rounded-lg">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-medium">Calinski-Harabasz Index</span>
                          <Badge variant="outline">568.92</Badge>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate better-defined clusters.
                        </p>
                      </div>
                    </div>
                  </Card>
                </div>

                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">3D Visualization</h4>
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=800"
                      alt="3D Clustering Visualization"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_iris_001.png"
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    For datasets with three features, a 3D visualization can provide additional insights into the cluster 
                    structure. This plot shows clusters in a three-dimensional space, with each axis representing a 
                    different feature. The clear separation between clusters in 3D space confirms the effectiveness of 
                    K-Means for this dataset.
                  </p>
                </Card>

                <div className="bg-muted/30 p-5 rounded-lg border">
                  <h4 className="font-medium text-lg mb-3">Interpreting Results</h4>
                  <div className="space-y-4">
                    <p className="text-sm text-muted-foreground">
                      When interpreting K-Means clustering results, consider the following aspects:
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-green-50 dark:bg-green-950 p-4 rounded-lg border border-green-100 dark:border-green-900">
                        <h5 className="font-medium mb-2">Cluster Centroids</h5>
                        <p className="text-sm text-green-700 dark:text-green-300">
                          Centroids represent the "average" of each cluster. Analyzing centroid values helps understand 
                          what makes each cluster unique. For example, a customer segment might have a centroid with high 
                          values for purchase frequency and low values for age.
                        </p>
                      </div>
                      <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-lg border border-blue-100 dark:border-blue-900">
                        <h5 className="font-medium mb-2">Cluster Size</h5>
                        <p className="text-sm text-blue-700 dark:text-blue-300">
                          The number of data points in each cluster indicates its prevalence. Highly imbalanced cluster 
                          sizes might suggest outliers or specialized segments. In marketing, a small but high-value 
                          customer segment might be particularly important.
                        </p>
                      </div>
                      <div className="bg-purple-50 dark:bg-purple-950 p-4 rounded-lg border border-purple-100 dark:border-purple-900">
                        <h5 className="font-medium mb-2">Cluster Separation</h5>
                        <p className="text-sm text-purple-700 dark:text-purple-300">
                          Well-separated clusters with minimal overlap indicate distinct groups. The silhouette score 
                          quantifies this separation. High separation suggests naturally occurring groups in the data, 
                          while low separation might indicate forced clustering.
                        </p>
                      </div>
                      <div className="bg-amber-50 dark:bg-amber-950 p-4 rounded-lg border border-amber-100 dark:border-amber-900">
                        <h5 className="font-medium mb-2">Domain Context</h5>
                        <p className="text-sm text-amber-700 dark:text-amber-300">
                          Always interpret clusters in the context of the domain. For example, in customer segmentation, 
                          clusters might represent different buying behaviors, while in image processing, they might 
                          represent different objects or regions.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
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
                <span>Implementing K-Means in Python is straightforward with scikit-learn</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Feature scaling is crucial for accurate distance calculations</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>The k-means++ initialization method helps find better starting centroids</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Visualizing clusters and analyzing centroids provides valuable insights</span>
              </li>
            </ul>
          </Card>
        </div>
      )
    }

    // Section 5: Visualization
    if (section === 5) {
      function copyToClipboard(visualizationCode: string, arg1: string): void {
        throw new Error("Function not implemented.")
      }

      return (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950 dark:to-purple-950 p-6 rounded-lg border border-indigo-100 dark:border-indigo-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-indigo-100 dark:bg-indigo-900 p-2 rounded-full">
                <PieChart className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
              </div>
              <h3 className="text-xl font-semibold text-indigo-800 dark:text-indigo-300">
                Visualization Techniques
              </h3>
            </div>
            <p className="text-indigo-700 dark:text-indigo-300 leading-relaxed">
              Visualizing K-Means clustering results is essential for understanding the structure of your data and 
              communicating insights effectively. This section covers various visualization techniques for different 
              dimensions and purposes, from basic scatter plots to advanced dimensionality reduction methods.
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="explanation">
                <BookOpen className="h-4 w-4 mr-2" />
                Techniques
              </TabsTrigger>
              <TabsTrigger value="code">
                <Code className="h-4 w-4 mr-2" />
                Implementation
              </TabsTrigger>
              <TabsTrigger value="visualization">
                <BarChart className="h-4 w-4 mr-2" />
                Examples
              </TabsTrigger>
            </TabsList>

            <TabsContent value="explanation" className="space-y-4 mt-4">
              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Visualization Approaches</h4>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      1
                    </div>
                    <div>
                      <h5 className="font-medium">Direct Visualization (2D/3D)</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        For datasets with 2 or 3 features, direct visualization using scatter plots is the most 
                        straightforward approach. Each point is colored according to its cluster assignment, and centroids 
                        are typically marked with distinct symbols.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      2
                    </div>
                    <div>
                      <h5 className="font-medium">Dimensionality Reduction</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        For high-dimensional data, techniques like PCA (Principal Component Analysis), t-SNE (t-Distributed 
                        Stochastic Neighbor Embedding), or UMAP (Uniform Manifold Approximation and Projection) can reduce 
                        the dimensions to 2D or 3D for visualization while preserving the structure.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      3
                    </div>
                    <div>
                      <h5 className="font-medium">Pair Plots</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Pair plots show pairwise relationships between features, with points colored by cluster. This 
                        provides multiple 2D views of the data, useful for understanding how clusters are separated across 
                        different feature combinations.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      4
                    </div>
                    <div>
                      <h5 className="font-medium">Cluster Profiles</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Visualizing the characteristics of each cluster using bar charts, radar charts, or parallel 
                        coordinate plots helps in understanding what makes each cluster unique.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-indigo-100 dark:bg-indigo-900 text-indigo-600 dark:text-indigo-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                      5
                    </div>
                    <div>
                      <h5 className="font-medium">Silhouette Plots</h5>
                      <p className="text-muted-foreground text-sm mt-1">
                        Silhouette plots visualize how well each point fits within its assigned cluster compared to other 
                        clusters, providing insights into cluster quality and separation.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Image className="h-5 w-5 text-indigo-500" />
                    <h4 className="font-medium">2D Visualization</h4>
                  </div>
                  <p className="text-muted-foreground mb-3">
                    2D scatter plots are the most common way to visualize clustering results when working with two features 
                    or after dimensionality reduction.
                  </p>
                  <div className="bg-muted/30 p-3 rounded-lg">
                    <h5 className="font-medium mb-1">Key Elements:</h5>
                    <ul className="text-sm text-muted-foreground space-y-1 list-disc pl-5">
                      <li>Points colored by cluster assignment</li>
                      <li>Centroids marked with distinct symbols</li>
                      <li>Clear axes labels and title</li>
                      <li>Legend to identify clusters</li>
                      <li>Optional decision boundaries</li>
                    </ul>
                  </div>
                  <div className="mt-3 bg-indigo-50 dark:bg-indigo-950 p-3 rounded-md">
                    <p className="text-xs text-indigo-700 dark:text-indigo-300">
                      <strong>Tip:</strong> Use alpha transparency (e.g., alpha=0.7) when plotting points to better 
                      visualize overlapping regions and density.
                    </p>
                  </div>
                </Card>

                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <GitBranch className="h-5 w-5 text-purple-500" />
                    <h4 className="font-medium">Dimensionality Reduction</h4>
                  </div>
                  <p className="text-muted-foreground mb-3">
                    For high-dimensional data, dimensionality reduction techniques help visualize the cluster structure.
                  </p>
                  <div className="space-y-3">
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <h5 className="font-medium mb-1">PCA</h5>
                      <p className="text-xs text-muted-foreground">
                        Linear technique that preserves global structure. Fast but may not capture non-linear relationships.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <h5 className="font-medium mb-1">t-SNE</h5>
                      <p className="text-xs text-muted-foreground">
                        Non-linear technique that preserves local structure. Good for visualizing clusters but 
                        computationally intensive.
                      </p>
                    </div>
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <h5 className="font-medium mb-1">UMAP</h5>
                      <p className="text-xs text-muted-foreground">
                        Modern alternative to t-SNE that preserves both local and global structure. Faster than t-SNE.
                      </p>
                    </div>
                  </div>
                </Card>
              </div>

              <Card className="p-5">
                <h4 className="font-medium text-lg mb-3">Advanced Visualization Techniques</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-muted/30 p-4 rounded-lg">
                    <h5 className="font-medium mb-2">Parallel Coordinates</h5>
                    <p className="text-sm text-muted-foreground">
                      Visualizes multiple dimensions by plotting each feature on a separate vertical axis and connecting 
                      points with lines. Lines are colored by cluster, revealing patterns across features.
                    </p>
                  </div>
                  <div className="bg-muted/30 p-4 rounded-lg">
                    <h5 className="font-medium mb-2">Radar Charts</h5>
                    <p className="text-sm text-muted-foreground">
                      Also known as spider charts, these display multivariate data as a two-dimensional chart with three or 
                      more quantitative variables. Each cluster's centroid can be plotted as a separate radar chart.
                    </p>
                  </div>
                  <div className="bg-muted/30 p-4 rounded-lg">
                    <h5 className="font-medium mb-2">Heatmaps</h5>
                    <p className="text-sm text-muted-foreground">
                      Visualizes feature values across clusters using color intensity. Rows represent data points (sorted 
                      by cluster), columns represent features, and colors represent values.
                    </p>
                  </div>
                </div>
                <div className="mt-4 bg-indigo-50 dark:bg-indigo-950 p-4 rounded-lg">
                  <h5 className="font-medium mb-2">Interactive Visualizations</h5>
                  <p className="text-sm text-indigo-700 dark:text-indigo-300">
                    For complex datasets, interactive visualizations using libraries like Plotly, Bokeh, or D3.js allow 
                    users to explore the data by zooming, panning, hovering for details, and filtering. This is 
                    particularly valuable for stakeholder presentations and exploratory analysis.
                  </p>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="code" className="mt-4">
              <Card className="p-5">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">2D Scatter Plot</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(visualizationCode, "2d-scatter")}
                    className="text-xs"
                  >
                    {copiedState === "2d-scatter" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{visualizationCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code creates a 2D scatter plot of the clustering results, with each cluster represented by a 
                    different color and centroids marked with yellow stars. This is the most basic and common visualization 
                    for K-Means clustering.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">PCA for Dimensionality Reduction</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(`import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Assuming X is your high-dimensional data
# and kmeans is your fitted K-Means model

# 1. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Get cluster labels
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 4. Transform centroids to PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)

# 5. Visualize the results
plt.figure(figsize=(12, 8))

# Plot each cluster with a different color
colors = ['red', 'blue', 'green', 'cyan']
for i in range(4):
    plt.scatter(X_pca[labels == i, 0], 
                X_pca[labels == i, 1], 
                s=100, c=colors[i], 
                label=f'Cluster {i+1}',
                alpha=0.7)

# Plot the centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            s=300, c='yellow', 
            marker='*', 
            label='Centroids',
            edgecolor='black')

# Add labels and legend
plt.title('K-Means Clustering with PCA', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Add explained variance information
plt.figtext(0.5, 0.01, f'Total explained variance: {sum(pca.explained_variance_ratio_):.2%}', 
           ha='center', fontsize=12)

plt.show()

# Print feature importance in PCA
print("Feature importance in principal components:")
for i, component in enumerate(pca.components_):
    print(f"PC{i+1}:", end=" ")
    # Get the indices of the top 3 features with highest absolute weights
    top_features = np.abs(component).argsort()[-3:][::-1]
    for idx in top_features:
        print(f"Feature {idx}: {component[idx]:.3f}", end=", ")
    print()`, "pca-code")}
                    className="text-xs"
                  >
                    {copiedState === "pca-code" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{`import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Assuming X is your high-dimensional data
# and kmeans is your fitted K-Means model

# 1. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Get cluster labels
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 4. Transform centroids to PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)

# 5. Visualize the results
plt.figure(figsize=(12, 8))

# Plot each cluster with a different color
colors = ['red', 'blue', 'green', 'cyan']
for i in range(4):
    plt.scatter(X_pca[labels == i, 0], 
                X_pca[labels == i, 1], 
                s=100, c=colors[i], 
                label=f'Cluster {i+1}',
                alpha=0.7)

# Plot the centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            s=300, c='yellow', 
            marker='*', 
            label='Centroids',
            edgecolor='black')

# Add labels and legend
plt.title('K-Means Clustering with PCA', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Add explained variance information
plt.figtext(0.5, 0.01, f'Total explained variance: {sum(pca.explained_variance_ratio_):.2%}', 
           ha='center', fontsize=12)

plt.show()

# Print feature importance in PCA
print("Feature importance in principal components:")
for i, component in enumerate(pca.components_):
    print(f"PC{i+1}:", end=" ")
    # Get the indices of the top 3 features with highest absolute weights
    top_features = np.abs(component).argsort()[-3:][::-1]
    for idx in top_features:
        print(f"Feature {idx}: {component[idx]:.3f}", end=", ")
    print()`}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code applies Principal Component Analysis (PCA) to reduce high-dimensional data to 2D for 
                    visualization. It also transforms the cluster centroids to the same PCA space and displays the 
                    explained variance ratio for each principal component.
                  </p>
                </div>
              </Card>

              <Card className="p-5 mt-4">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Silhouette Plot</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(silhouetteCode, "silhouette-plot")}
                    className="text-xs"
                  >
                    {copiedState === "silhouette-plot" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{silhouetteCode}</code>
                </pre>
                <div className="mt-3 text-sm text-muted-foreground">
                  <p>
                    This code creates a silhouette plot to visualize how well each data point fits within its assigned 
                    cluster. The silhouette coefficient ranges from -1 to 1, with higher values indicating better cluster 
                    assignment. This visualization helps in assessing the quality of clustering.
                  </p>
                </div>
              </Card>

              <div className="bg-indigo-50 dark:bg-indigo-950 p-4 rounded-lg border border-indigo-100 dark:border-indigo-900 mt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb className="h-5 w-5 text-indigo-500" />
                  <h4 className="font-medium">Advanced Visualization Example</h4>
                </div>
                <p className="text-sm text-indigo-700 dark:text-indigo-300 mb-3">
                  Here's an example of creating a parallel coordinates plot to visualize cluster profiles:
                </p>
                <pre className="bg-indigo-100/50 dark:bg-indigo-900/50 p-3 rounded-md text-xs overflow-x-auto">
                  <code>{`import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Assuming X is your data, feature_names are your column names,
# and labels are your cluster assignments

# Create a DataFrame with features and cluster labels
df = pd.DataFrame(X, columns=feature_names)
df['Cluster'] = labels

# Standardize the features for better visualization
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df.drop('Cluster', axis=1)),
    columns=feature_names
)
df_scaled['Cluster'] = df['Cluster']

# Convert cluster labels to strings for better visualization
df_scaled['Cluster'] = 'Cluster ' + df_scaled['Cluster'].astype(str)

# Create parallel coordinates plot
fig = px.parallel_coordinates(
    df_scaled, 
    color="Cluster",
    dimensions=feature_names,
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="Parallel Coordinates Plot of Clusters"
)

# Update layout
fig.update_layout(
    font=dict(size=12),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Show the plot
fig.show()

# For static matplotlib version (less interactive but works in all environments)
from pandas.plotting import parallel_coordinates

plt.figure(figsize=(15, 8))
parallel_coordinates(df_scaled, 'Cluster', colormap='viridis')
plt.title('Parallel Coordinates Plot of Clusters', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()`}</code>
                </pre>
                <p className="text-sm text-indigo-700 dark:text-indigo-300 mt-3">
                  This code creates a parallel coordinates plot where each line represents a data point, colored by its 
                  cluster assignment. This visualization is particularly useful for understanding how clusters differ 
                  across multiple features simultaneously.
                </p>
              </div>
            </TabsContent>

            <TabsContent value="visualization" className="mt-4">
              <div className="space-y-6">
                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">2D Scatter Plot</h4>
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=800"
                      alt="2D Scatter Plot"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_assumptions_001.png"
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    This 2D scatter plot shows data points colored by their cluster assignments, with centroids marked as 
                    stars. This is the most basic visualization for K-Means clustering and works well for datasets with two 
                    features or after dimensionality reduction.
                  </p>
                </Card>

                <Card className="p-5">
                  <h4 className="font-medium text-lg mb-3">Silhouette Plot</h4>
                  <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                    <img
                      src="/placeholder.svg?height=400&width=800"
                      alt="Silhouette Plot"
                      className="max-w-full h-auto"
                      onError={(e) => {
                        e.currentTarget.src = "https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_silhouette_001.png"
                      }}
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    This silhouette plot visualizes the quality of clustering by assigning a silhouette coefficient to each 
                    data point. This coefficient ranges from -1 to 1, with higher values indicating better cluster 
                    assignment. This visualization helps in assessing the quality of clustering.
                  </p>
                </Card>
              </div>
              </TabsContent>
          </Tabs>
        </div>
      )
    }
    return (
      <div className="py-8 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <BookOpen className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-xl font-medium mb-2">Section Content Coming Soon</h3>
        <p className="text-muted-foreground max-w-md mx-auto">
          We're currently developing content for this section of the ensemble learning tutorial. Check back soon!
        </p>
      </div>
    )
  }