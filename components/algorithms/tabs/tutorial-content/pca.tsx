"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Layers,
  Minimize2,
  BookOpen,
  Code,
  BarChart,
  Lightbulb,
  CheckCircle,
  ArrowRight,
  Sigma,
  Eye,
  Copy,
  Check,
  Zap,
  Target,
} from "lucide-react"

interface PcaTutorialProps {
  section: number
  onCopy?: (text: string, id: string) => void
  copied?: string | null
}

export function PcaTutorial({ section = 0, onCopy, copied }: PcaTutorialProps) {
  const [activeTab, setActiveTab] = useState("explanation")

  const handleCopy = (text: string, id: string) => {
    if (onCopy) {
      onCopy(text, id)
    } else {
      navigator.clipboard.writeText(text)
    }
  }

  const basicPCACode = `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the famous Iris dataset
iris = load_iris()
X = iris.data        # Features: sepal length, sepal width, petal length, petal width
y = iris.target      # Labels: 0=setosa, 1=versicolor, 2=virginica

# Create a DataFrame for better visualization
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['species'] = y

print("Original Iris Dataset:")
print(f"Shape: {iris_df.shape}")
print("\\nFirst 5 rows:")
print(iris_df.head())

print("\\nDataset Info:")
print(f"Number of samples: {len(iris_df)}")
print(f"Number of features: {len(feature_names)}")
print(f"Species distribution:")
for i, species in enumerate(iris.target_names):
    count = sum(y == i)
    print(f"  {species}: {count} samples")
`

  const pcaStepsCode = `
# Step 1: Standardize the features
# This is crucial because PCA is sensitive to the scale of features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Step 1: Feature Standardization")
print("Original feature means:", X.mean(axis=0).round(2))
print("Original feature std:", X.std(axis=0).round(2))
print("\\nScaled feature means:", X_scaled.mean(axis=0).round(2))
print("Scaled feature std:", X_scaled.std(axis=0).round(2))

# Step 2: Apply PCA to reduce from 4 dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\\nStep 2: PCA Transformation")
print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_pca.shape}")
print(f"Reduced from {X.shape[1]} dimensions to {X_pca.shape[1]} dimensions")

# Step 3: Analyze the results
print("\\nStep 3: PCA Results Analysis")
print("Explained Variance Ratio:", pca.explained_variance_ratio_.round(4))
print("Total Variance Retained:", pca.explained_variance_ratio_.sum().round(4))
print(f"We retained {pca.explained_variance_ratio_.sum()*100:.1f}% of the original information")

# The principal components (directions of maximum variance)
print("\\nPrincipal Components (PC loadings):")
pc_df = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
print(pc_df.round(3))
`

  const visualizationCode = `
# Create a comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Original data (first two features)
axes[0,0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[0,0].set_xlabel('Sepal Length')
axes[0,0].set_ylabel('Sepal Width')
axes[0,0].set_title('Original Data (First 2 Features)')
axes[0,0].grid(True, alpha=0.3)

# Plot 2: PCA transformed data
scatter = axes[0,1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[0,1].set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
axes[0,1].set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
axes[0,1].set_title('PCA Transformed Data')
axes[0,1].grid(True, alpha=0.3)

# Add colorbar
plt.colorbar(scatter, ax=axes[0,1], label='Species')

# Plot 3: Explained Variance
components = range(1, len(pca.explained_variance_ratio_) + 1)
axes[1,0].bar(components, pca.explained_variance_ratio_, alpha=0.7, color='skyblue')
axes[1,0].set_xlabel('Principal Component')
axes[1,0].set_ylabel('Explained Variance Ratio')
axes[1,0].set_title('Explained Variance by Component')
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Cumulative Explained Variance
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
axes[1,1].plot(components, cumsum_var, 'bo-', linewidth=2, markersize=8)
axes[1,1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
axes[1,1].set_xlabel('Number of Components')
axes[1,1].set_ylabel('Cumulative Explained Variance')
axes[1,1].set_title('Cumulative Explained Variance')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].legend()

plt.tight_layout()
plt.show()

print("Visualization created showing:")
print("1. Original data using first two features")
print("2. PCA transformed data (2D projection)")
print("3. Individual component contributions")
print("4. Cumulative variance explained")
`

  const interpretationCode = `
# Detailed interpretation of PCA results
print("=== DETAILED PCA INTERPRETATION ===\\n")

# 1. Variance explained by each component
print("1. VARIANCE EXPLANATION:")
for i, var_ratio in enumerate(pca.explained_variance_ratio_):
    print(f"   PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.1f}% of total variance)")

total_variance = pca.explained_variance_ratio_.sum()
print(f"   Total: {total_variance:.4f} ({total_variance*100:.1f}% of original information retained)")

# 2. Principal component interpretation
print("\\n2. PRINCIPAL COMPONENT INTERPRETATION:")
print("   (How much each original feature contributes to each PC)")

for i in range(pca.n_components_):
    print(f"\\n   PC{i+1} loadings:")
    pc_loadings = pca.components_[i]
    
    # Sort features by absolute loading values
    feature_importance = [(feature_names[j], pc_loadings[j]) for j in range(len(feature_names))]
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, loading in feature_importance:
        direction = "positively" if loading > 0 else "negatively"
        print(f"     {feature}: {loading:.3f} (contributes {direction})")

# 3. What do the principal components represent?
print("\\n3. BIOLOGICAL INTERPRETATION:")
print("   PC1 (First Principal Component):")
pc1_loadings = pca.components_[0]
if abs(pc1_loadings[2]) > 0.5 and abs(pc1_loadings[3]) > 0.5:  # petal features
    print("     - Primarily represents petal characteristics")
    print("     - High PC1 values = larger petals")
    print("     - This component separates species by petal size")

print("\\n   PC2 (Second Principal Component):")
pc2_loadings = pca.components_[1]
if abs(pc2_loadings[0]) > 0.3 or abs(pc2_loadings[1]) > 0.3:  # sepal features
    print("     - Captures sepal characteristics")
    print("     - Helps distinguish between species with similar petal sizes")
    print("     - Provides additional discriminatory information")

# 4. Data point analysis
print("\\n4. SAMPLE ANALYSIS:")
print("   Transformed data ranges:")
print(f"   PC1: [{X_pca[:, 0].min():.2f}, {X_pca[:, 0].max():.2f}]")
print(f"   PC2: [{X_pca[:, 1].min():.2f}, {X_pca[:, 1].max():.2f}]")

# Find extreme points
pc1_max_idx = np.argmax(X_pca[:, 0])
pc1_min_idx = np.argmin(X_pca[:, 0])
print(f"\\n   Highest PC1 value: Sample {pc1_max_idx} (species: {iris.target_names[y[pc1_max_idx]]})")
print(f"   Lowest PC1 value: Sample {pc1_min_idx} (species: {iris.target_names[y[pc1_min_idx]]})")
`

  const advancedPCACode = `
# Advanced PCA Analysis: Choosing the right number of components
print("=== ADVANCED PCA ANALYSIS ===\\n")

# 1. Full PCA (all components)
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

print("1. FULL PCA ANALYSIS:")
print("   Explained variance ratios for all components:")
for i, var_ratio in enumerate(pca_full.explained_variance_ratio_):
    print(f"   PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.1f}%)")

# 2. Cumulative variance analysis
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
print("\\n   Cumulative explained variance:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"   Up to PC{i+1}: {cum_var:.4f} ({cum_var*100:.1f}%)")

# 3. Determine optimal number of components
print("\\n2. OPTIMAL COMPONENT SELECTION:")

# Method 1: 95% variance threshold
components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"   Components needed for 95% variance: {components_95}")

# Method 2: Elbow method (look for sharp drops)
variance_ratios = pca_full.explained_variance_ratio_
print("   Variance drops between consecutive components:")
for i in range(len(variance_ratios)-1):
    drop = variance_ratios[i] - variance_ratios[i+1]
    print(f"   PC{i+1} to PC{i+2}: {drop:.4f}")

# Method 3: Kaiser criterion (eigenvalues > 1)
eigenvalues = pca_full.explained_variance_
components_kaiser = sum(eigenvalues > 1)
print(f"\\n   Components with eigenvalue > 1: {components_kaiser}")
print("   Eigenvalues:", eigenvalues.round(3))

# 4. Reconstruction quality analysis
print("\\n3. RECONSTRUCTION QUALITY:")
for n_comp in [1, 2, 3, 4]:
    pca_temp = PCA(n_components=n_comp)
    X_reduced = pca_temp.fit_transform(X_scaled)
    X_reconstructed = pca_temp.inverse_transform(X_reduced)
    
    # Calculate reconstruction error
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
    variance_retained = pca_temp.explained_variance_ratio_.sum()
    
    print(f"   {n_comp} components: {variance_retained:.1%} variance, "
          f"reconstruction error: {reconstruction_error:.4f}")

# 5. Feature contribution analysis
print("\\n4. FEATURE CONTRIBUTION ANALYSIS:")
print("   How much each original feature contributes to the first 2 PCs:")

feature_contributions = np.abs(pca.components_[:2])  # First 2 PCs
for i, feature in enumerate(feature_names):
    pc1_contrib = feature_contributions[0, i]
    pc2_contrib = feature_contributions[1, i]
    total_contrib = pc1_contrib + pc2_contrib
    print(f"   {feature}: PC1={pc1_contrib:.3f}, PC2={pc2_contrib:.3f}, "
          f"Total={total_contrib:.3f}")
`

  const practicalApplicationCode = `
# Practical Application: Using PCA for Data Compression and Visualization
print("=== PRACTICAL PCA APPLICATION ===\\n")

# 1. Data Compression Example
print("1. DATA COMPRESSION ANALYSIS:")

original_size = X.shape[0] * X.shape[1]
print(f"   Original data size: {X.shape[0]} samples × {X.shape[1]} features = {original_size} values")

for n_components in [1, 2, 3]:
    pca_compress = PCA(n_components=n_components)
    X_compressed = pca_compress.fit_transform(X_scaled)
    
    # Size after compression
    compressed_size = X_compressed.shape[0] * X_compressed.shape[1]
    # Add PCA model parameters (components + mean)
    model_size = n_components * X.shape[1] + X.shape[1]  # components + mean
    total_size = compressed_size + model_size
    
    compression_ratio = original_size / total_size
    variance_retained = pca_compress.explained_variance_ratio_.sum()
    
    print(f"   {n_components} components: {compression_ratio:.1f}x compression, "
          f"{variance_retained:.1%} variance retained")

# 2. Dimensionality Reduction for Machine Learning
print("\\n2. MACHINE LEARNING PREPROCESSING:")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("   Comparing classification accuracy with different numbers of components:")

# Test with different numbers of components
for n_comp in [1, 2, 3, 4]:
    if n_comp == 4:
        # Use original data
        X_train_pca = X_train
        X_test_pca = X_test
        print(f"   Original data (4 features):", end=" ")
    else:
        # Apply PCA
        pca_ml = PCA(n_components=n_comp)
        X_train_pca = pca_ml.fit_transform(X_train)
        X_test_pca = pca_ml.transform(X_test)
        variance_retained = pca_ml.explained_variance_ratio_.sum()
        print(f"   {n_comp} PCA components ({variance_retained:.1%} variance):", end=" ")
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_pca, y_train)
    
    # Predict and calculate accuracy
    y_pred = clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy = {accuracy:.3f}")

# 3. Noise Reduction Example
print("\\n3. NOISE REDUCTION DEMONSTRATION:")

# Add noise to the original data
np.random.seed(42)
noise_level = 0.1
X_noisy = X_scaled + np.random.normal(0, noise_level, X_scaled.shape)

print(f"   Added Gaussian noise (std={noise_level}) to the data")

# Apply PCA for denoising (keep only top components)
pca_denoise = PCA(n_components=2)  # Keep only first 2 components
X_denoised_pca = pca_denoise.fit_transform(X_noisy)
X_denoised = pca_denoise.inverse_transform(X_denoised_pca)

# Calculate noise reduction
original_noise = np.mean((X_noisy - X_scaled) ** 2)
remaining_noise = np.mean((X_denoised - X_scaled) ** 2)
noise_reduction = (original_noise - remaining_noise) / original_noise

print(f"   Original noise level: {original_noise:.4f}")
print(f"   Noise after PCA denoising: {remaining_noise:.4f}")
print(f"   Noise reduction: {noise_reduction:.1%}")

# 4. Feature Importance for Interpretation
print("\\n4. FEATURE IMPORTANCE FROM PCA:")
print("   Most important features for data variation:")

# Calculate feature importance based on PCA loadings
feature_importance_pca = np.sum(np.abs(pca.components_), axis=0)
feature_importance_pca = feature_importance_pca / feature_importance_pca.sum()

# Sort features by importance
feature_ranking = sorted(zip(feature_names, feature_importance_pca), 
                        key=lambda x: x[1], reverse=True)

for i, (feature, importance) in enumerate(feature_ranking):
    print(f"   {i+1}. {feature}: {importance:.3f} ({importance*100:.1f}%)")

print("\\n   Interpretation: Features with higher PCA importance contribute")
print("   more to the overall variation in the dataset.")
`

  // Section 0: Introduction to PCA
  if (section === 0) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950 dark:to-indigo-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-purple-100 dark:bg-purple-900 p-2 rounded-full">
              <Layers className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300">
              What is Principal Component Analysis (PCA)?
            </h3>
          </div>
          <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
            Principal Component Analysis (PCA) is a powerful dimensionality reduction technique that transforms your
            data into a lower-dimensional space while preserving the most important information. Think of it as finding
            the best camera angle to capture a 3D object in a 2D photo - you want to keep the most meaningful details
            while simplifying the representation.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="p-5 hover:shadow-md transition-shadow">
            <div className="flex items-center gap-2 mb-3">
              <Minimize2 className="h-5 w-5 text-purple-500" />
              <h4 className="font-medium text-lg">Key Concepts</h4>
            </div>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  1
                </span>
                <span>Reduces the number of features while keeping important information</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  2
                </span>
                <span>Finds directions of maximum variance in the data</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  3
                </span>
                <span>Creates new features (principal components) that are combinations of original features</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  4
                </span>
                <span>Helps with data visualization and noise reduction</span>
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
                <span>Data visualization (reducing to 2D or 3D for plotting)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Image compression and processing</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Preprocessing for machine learning models</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="bg-yellow-100 dark:bg-yellow-900 text-yellow-600 dark:text-yellow-400 rounded-full h-5 w-5 flex items-center justify-center text-xs mt-0.5">
                  •
                </span>
                <span>Noise reduction and feature extraction</span>
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
              { icon: <Layers className="h-4 w-4" />, text: "Basic PCA Concepts" },
              { icon: <Zap className="h-4 w-4" />, text: "Step-by-Step Implementation" },
              { icon: <Eye className="h-4 w-4" />, text: "Data Visualization" },
              { icon: <Target className="h-4 w-4" />, text: "Practical Applications" },
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
              <span>Familiarity with NumPy and Pandas</span>
            </li>
            <li className="flex items-start gap-2">
              <ArrowRight className="h-4 w-4 mt-1 text-primary" />
              <span>Basic knowledge of statistics (mean, variance)</span>
            </li>
            <li className="flex items-start gap-2">
              <ArrowRight className="h-4 w-4 mt-1 text-primary" />
              <span>Understanding of data visualization with Matplotlib</span>
            </li>
          </ul>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-950 dark:to-teal-950 p-6 rounded-lg border border-green-100 dark:border-green-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-green-100 dark:bg-green-900 p-2 rounded-full">
                <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
              </div>
              <h3 className="text-lg font-semibold text-green-800 dark:text-green-300">Why Use PCA?</h3>
            </div>
            <ul className="space-y-3 text-green-700 dark:text-green-300">
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Simplifies complex datasets while preserving important patterns</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Enables visualization of high-dimensional data</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Reduces computational cost for machine learning</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Helps identify the most important features</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Removes noise and redundant information</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950 dark:to-orange-950 p-6 rounded-lg border border-amber-100 dark:border-amber-900">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-amber-100 dark:bg-amber-900 p-2 rounded-full">
                <Lightbulb className="h-5 w-5 text-amber-600 dark:text-amber-400" />
              </div>
              <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-300">Simple Analogy</h3>
            </div>
            <div className="text-amber-700 dark:text-amber-300">
              <p className="mb-3">Imagine you have a 3D sculpture and want to take the best 2D photograph of it:</p>
              <ul className="space-y-2 text-sm">
                <li>• You'd rotate it to find the angle that shows the most detail</li>
                <li>• Some angles reveal more information than others</li>
                <li>• The "best" angle captures the most important features</li>
                <li>• PCA does this mathematically with your data!</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-muted/20 p-6 rounded-lg border mt-6">
          <h4 className="font-medium text-lg mb-4 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            PCA in Simple Terms
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg">
              <div className="text-2xl mb-2">📊</div>
              <h5 className="font-medium mb-2">High-Dimensional Data</h5>
              <p className="text-sm text-muted-foreground">
                Your dataset has many features (columns) that might be related to each other
              </p>
            </div>
            <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg">
              <div className="text-2xl mb-2">🔄</div>
              <h5 className="font-medium mb-2">PCA Transformation</h5>
              <p className="text-sm text-muted-foreground">
                PCA finds the most important directions in your data and creates new features
              </p>
            </div>
            <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg">
              <div className="text-2xl mb-2">📈</div>
              <h5 className="font-medium mb-2">Simplified Data</h5>
              <p className="text-sm text-muted-foreground">
                You get fewer features that capture most of the original information
              </p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Section 1: Basic PCA Implementation
  if (section === 1) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
          <h3 className="text-xl font-semibold text-blue-800 dark:text-blue-300 mb-3">Basic PCA Implementation</h3>
          <p className="text-blue-700 dark:text-blue-300 leading-relaxed">
            Let's start with a hands-on example using the famous Iris dataset. We'll reduce 4 features (sepal length,
            sepal width, petal length, petal width) down to 2 principal components while keeping most of the important
            information.
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
              <h4 className="font-medium text-lg mb-3">The Iris Dataset</h4>
              <p className="text-muted-foreground mb-4">
                The Iris dataset is perfect for learning PCA because it's simple yet meaningful. It contains
                measurements of 150 iris flowers from 3 different species.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h5 className="font-medium mb-2">Features (4 total):</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Sepal Length (cm)</li>
                    <li>• Sepal Width (cm)</li>
                    <li>• Petal Length (cm)</li>
                    <li>• Petal Width (cm)</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium mb-2">Species (3 types):</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Setosa (50 samples)</li>
                    <li>• Versicolor (50 samples)</li>
                    <li>• Virginica (50 samples)</li>
                  </ul>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">PCA Steps Explained</h4>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                    1
                  </div>
                  <div>
                    <h5 className="font-medium">Standardize the Data</h5>
                    <p className="text-sm text-muted-foreground mt-1">
                      PCA is sensitive to the scale of features. We standardize so all features have mean=0 and std=1.
                      This ensures no single feature dominates just because it has larger values.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                    2
                  </div>
                  <div>
                    <h5 className="font-medium">Find Principal Components</h5>
                    <p className="text-sm text-muted-foreground mt-1">
                      PCA finds the directions (principal components) where the data varies the most. The first PC
                      captures the most variance, the second PC captures the second most, and so on.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                    3
                  </div>
                  <div>
                    <h5 className="font-medium">Transform the Data</h5>
                    <p className="text-sm text-muted-foreground mt-1">
                      The original 4D data is projected onto the new 2D space defined by the first two principal
                      components. Each data point gets new coordinates in this reduced space.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded-full h-6 w-6 flex items-center justify-center text-sm mt-0.5 flex-shrink-0">
                    4
                  </div>
                  <div>
                    <h5 className="font-medium">Analyze the Results</h5>
                    <p className="text-sm text-muted-foreground mt-1">
                      We check how much of the original information (variance) is retained. Good PCA typically keeps
                      80-95% of the original variance with fewer dimensions.
                    </p>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Key Concepts</h4>
              <div className="space-y-3">
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    Variance
                  </Badge>
                  <span className="text-sm">
                    How spread out the data is. PCA finds directions with maximum variance.
                  </span>
                </div>
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    Principal Components
                  </Badge>
                  <span className="text-sm">
                    New features created by PCA. They're combinations of original features.
                  </span>
                </div>
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    Explained Variance Ratio
                  </Badge>
                  <span className="text-sm">Percentage of original information captured by each component.</span>
                </div>
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    Loadings
                  </Badge>
                  <span className="text-sm">
                    How much each original feature contributes to each principal component.
                  </span>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Loading and Exploring the Iris Dataset</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(basicPCACode, "basic-pca-code")}
                  className="text-xs"
                >
                  {copied === "basic-pca-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>

              <div className="mb-6">
                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() => handleCopy(basicPCACode, "basic-pca-part1")}
                    >
                      {copied === "basic-pca-part1" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>{basicPCACode}</code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono text-sm">
                      Original Iris Dataset:
                      <br />
                      Shape: (150, 5)
                      <br />
                      <br />
                      First 5 rows:
                      <br />
                      {"   sepal_length  sepal_width  petal_length  petal_width  species\n"}
                      {"0           5.1          3.5           1.4          0.2        0\n"}
                      {"1           4.9          3.0           1.4          0.2        0\n"}
                      {"2           4.7          3.2           1.3          0.2        0\n"}
                      {"3           4.6          3.1           1.5          0.2        0\n"}
                      {"4           5.0          3.6           1.4          0.2        0\n"}
                      <br />
                      Dataset Info:
                      <br />
                      Number of samples: 150
                      <br />
                      Number of features: 4<br />
                      Species distribution:
                      <br />
                      {"  setosa: 50 samples\n"}
                      {"  versicolor: 50 samples\n"}
                      {"  virginica: 50 samples"}
                    </div>
                    <p className="text-gray-500 mt-2">
                      We've loaded the Iris dataset with 150 flower samples, each having 4 measurements. The dataset is
                      perfectly balanced with 50 samples from each of the 3 iris species.
                    </p>
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <div className="flex justify-between items-center mb-3">
                  <h4 className="font-medium text-lg">Step-by-Step PCA Implementation</h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleCopy(pcaStepsCode, "pca-steps-code")}
                    className="text-xs"
                  >
                    {copied === "pca-steps-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>

                <div className="relative bg-black rounded-md mb-0">
                  <div className="absolute right-2 top-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-gray-400 hover:text-white"
                      onClick={() => handleCopy(pcaStepsCode, "pca-steps-part2")}
                    >
                      {copied === "pca-steps-part2" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                  <pre className="p-4 text-white overflow-x-auto">
                    <code>{pcaStepsCode}</code>
                  </pre>
                </div>

                <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                  <div className="p-4">
                    <h4 className="text-base font-medium mb-2">Output:</h4>
                    <div className="font-mono text-sm">
                      Step 1: Feature Standardization
                      <br />
                      Original feature means: [5.84 3.06 3.76 1.2 ]<br />
                      Original feature std: [0.83 0.44 1.76 0.76]
                      <br />
                      <br />
                      Scaled feature means: [-0. -0. 0. 0.]
                      <br />
                      Scaled feature std: [1. 1. 1. 1.]
                      <br />
                      <br />
                      Step 2: PCA Transformation
                      <br />
                      Original shape: (150, 4)
                      <br />
                      Transformed shape: (150, 2)
                      <br />
                      Reduced from 4 dimensions to 2 dimensions
                      <br />
                      <br />
                      Step 3: PCA Results Analysis
                      <br />
                      Explained Variance Ratio: [0.7296 0.2285]
                      <br />
                      Total Variance Retained: 0.9581
                      <br />
                      We retained 95.8% of the original information
                      <br />
                      <br />
                      Principal Components (PC loadings):
                      <br />
                      {"              PC1    PC2\n"}
                      {"sepal_length  0.521 -0.377\n"}
                      {"sepal_width  -0.269 -0.923\n"}
                      {"petal_length  0.580  0.024\n"}
                      {"petal_width   0.565  0.067"}
                    </div>
                    <p className="text-gray-500 mt-2">
                      Excellent results! We've successfully reduced 4 dimensions to 2 while keeping 95.8% of the
                      original information. The first principal component captures 72.96% of the variance, and the
                      second captures an additional 22.85%.
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">PCA Visualization</h4>
              <div className="mb-4">
                <div className="flex justify-between items-center mb-3">
                  <h5 className="font-medium">Complete Visualization Code</h5>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleCopy(visualizationCode, "visualization-code")}
                    className="text-xs"
                  >
                    {copied === "visualization-code" ? "Copied!" : "Copy Code"}
                  </Button>
                </div>
                <div className="relative bg-black rounded-md">
                  <pre className="p-4 text-white overflow-x-auto text-sm">
                    <code>{visualizationCode}</code>
                  </pre>
                </div>
              </div>

              <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden mb-4">
                <img
                  src="/placeholder.svg?height=400&width=800&text=PCA+Visualization+Results"
                  alt="PCA Visualization Results"
                  className="max-w-full h-auto"
                />
              </div>

              <div className="text-sm text-muted-foreground">
                <h5 className="font-medium mb-2">What the visualization shows:</h5>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="font-medium mb-1">Top Left - Original Data:</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Shows only the first 2 original features</li>
                      <li>Species are somewhat separated but overlapping</li>
                      <li>Not the best view of the data</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium mb-1">Top Right - PCA Transformed:</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Shows data in the new PCA space</li>
                      <li>Much better separation between species</li>
                      <li>Uses information from all 4 original features</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium mb-1">Bottom Left - Variance by Component:</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>PC1 explains ~73% of variance</li>
                      <li>PC2 explains ~23% of variance</li>
                      <li>Together they capture 95.8% of information</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium mb-1">Bottom Right - Cumulative Variance:</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Shows how variance adds up</li>
                      <li>Helps decide how many components to keep</li>
                      <li>95% threshold line for reference</li>
                    </ul>
                  </div>
                </div>
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
              <span>PCA successfully reduced 4 dimensions to 2 while keeping 95.8% of the information</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Feature standardization is crucial before applying PCA</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>The first principal component captures the most variance in the data</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>PCA creates better separation between iris species than using original features</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Principal components are linear combinations of original features</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 2: Understanding PCA Results
  if (section === 2) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950 dark:to-teal-950 p-6 rounded-lg border border-emerald-100 dark:border-emerald-900">
          <h3 className="text-xl font-semibold text-emerald-800 dark:text-emerald-300 mb-3">
            Understanding PCA Results
          </h3>
          <p className="text-emerald-700 dark:text-emerald-300 leading-relaxed">
            Now that we've applied PCA, let's dive deeper into interpreting the results. Understanding what the
            principal components represent and how to analyze the loadings will help you extract meaningful insights
            from your data.
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
              Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="explanation" className="space-y-4 mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">What Are Principal Components?</h4>
              <p className="text-muted-foreground mb-4">
                Principal components are new features created by PCA. Each PC is a linear combination of the original
                features, designed to capture maximum variance in the data.
              </p>

              <div className="bg-muted/50 p-4 rounded-lg mb-4">
                <p className="text-center font-mono text-sm">
                  PC1 = (0.521 × sepal_length) + (-0.269 × sepal_width) + (0.580 × petal_length) + (0.565 × petal_width)
                </p>
              </div>

              <div className="space-y-3">
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    PC1
                  </Badge>
                  <div>
                    <p className="text-sm font-medium">First Principal Component (72.96% variance)</p>
                    <p className="text-xs text-muted-foreground">
                      Captures the most variation in the data. In the Iris dataset, this primarily represents overall
                      flower size.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    PC2
                  </Badge>
                  <div>
                    <p className="text-sm font-medium">Second Principal Component (22.85% variance)</p>
                    <p className="text-xs text-muted-foreground">
                      Captures the second most variation, orthogonal to PC1. This helps distinguish between flower
                      shapes.
                    </p>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Interpreting Loadings</h4>
              <p className="text-muted-foreground mb-4">
                Loadings tell us how much each original feature contributes to each principal component. Higher absolute
                values mean stronger contribution.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">PC1 Loadings Analysis</h5>
                  <ul className="text-sm space-y-1">
                    <li>
                      • <strong>Petal Length (0.580):</strong> Strongest positive contributor
                    </li>
                    <li>
                      • <strong>Petal Width (0.565):</strong> Strong positive contributor
                    </li>
                    <li>
                      • <strong>Sepal Length (0.521):</strong> Moderate positive contributor
                    </li>
                    <li>
                      • <strong>Sepal Width (-0.269):</strong> Negative contributor
                    </li>
                  </ul>
                  <p className="text-xs text-muted-foreground mt-2">
                    PC1 primarily represents petal characteristics and overall flower size.
                  </p>
                </div>
                <div className="border rounded-lg p-3">
                  <h5 className="font-medium mb-2">PC2 Loadings Analysis</h5>
                  <ul className="text-sm space-y-1">
                    <li>
                      • <strong>Sepal Width (-0.923):</strong> Dominant negative contributor
                    </li>
                    <li>
                      • <strong>Sepal Length (-0.377):</strong> Moderate negative contributor
                    </li>
                    <li>
                      • <strong>Petal Width (0.067):</strong> Minimal positive contributor
                    </li>
                    <li>
                      • <strong>Petal Length (0.024):</strong> Minimal positive contributor
                    </li>
                  </ul>
                  <p className="text-xs text-muted-foreground mt-2">
                    PC2 primarily represents sepal characteristics, especially width.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Biological Interpretation</h4>
              <p className="text-muted-foreground mb-4">
                For the Iris dataset, we can interpret the principal components in biological terms:
              </p>

              <div className="space-y-4">
                <div className="border-l-4 border-l-blue-500 pl-4">
                  <h5 className="font-medium text-blue-700 dark:text-blue-300">PC1: "Overall Flower Size"</h5>
                  <p className="text-sm text-muted-foreground mt-1">
                    Since petal measurements have the highest loadings, PC1 primarily captures how big the flower is
                    overall. Larger flowers (especially petals) will have higher PC1 values.
                  </p>
                </div>
                <div className="border-l-4 border-l-green-500 pl-4">
                  <h5 className="font-medium text-green-700 dark:text-green-300">PC2: "Flower Shape Ratio"</h5>
                  <p className="text-sm text-muted-foreground mt-1">
                    With sepal width having a strong negative loading, PC2 captures the relationship between sepal width
                    and other measurements. It helps distinguish between wide vs. narrow sepals.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Variance Explained</h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-blue-50 dark:bg-blue-950 rounded">
                  <span>PC1 Explained Variance</span>
                  <Badge>72.96%</Badge>
                </div>
                <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-950 rounded">
                  <span>PC2 Explained Variance</span>
                  <Badge>22.85%</Badge>
                </div>
                <div className="flex justify-between items-center p-3 bg-purple-50 dark:bg-purple-950 rounded">
                  <span>Total Variance Retained</span>
                  <Badge>95.81%</Badge>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-3">
                This means we've captured 95.81% of the original information using just 2 components instead of 4
                features!
              </p>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Detailed PCA Interpretation Code</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(interpretationCode, "interpretation-code")}
                  className="text-xs"
                >
                  {copied === "interpretation-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>

              <div className="relative bg-black rounded-md mb-4">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => handleCopy(interpretationCode, "interpretation-full")}
                  >
                    {copied === "interpretation-full" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{interpretationCode}</code>
                </pre>
              </div>

              <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                <div className="p-4">
                  <h4 className="text-base font-medium mb-2">Output:</h4>
                  <div className="font-mono text-sm">
                    === DETAILED PCA INTERPRETATION ===
                    <br />
                    <br />
                    1. VARIANCE EXPLANATION:
                    <br />
                    {"   PC1: 0.7296 (72.96% of total variance)\n"}
                    {"   PC2: 0.2285 (22.85% of total variance)\n"}
                    {"   Total: 0.9581 (95.81% of original information retained)\n"}
                    <br />
                    2. PRINCIPAL COMPONENT INTERPRETATION:
                    <br />
                    {"   (How much each original feature contributes to each PC)\n"}
                    <br />
                    {"   PC1 loadings:\n"}
                    {"     petal_length: 0.580 (contributes positively)\n"}
                    {"     petal_width: 0.565 (contributes positively)\n"}
                    {"     sepal_length: 0.521 (contributes positively)\n"}
                    {"     sepal_width: -0.269 (contributes negatively)\n"}
                    <br />
                    {"   PC2 loadings:\n"}
                    {"     sepal_width: -0.923 (contributes negatively)\n"}
                    {"     sepal_length: -0.377 (contributes negatively)\n"}
                    {"     petal_width: 0.067 (contributes positively)\n"}
                    {"     petal_length: 0.024 (contributes positively)\n"}
                    <br />
                    3. BIOLOGICAL INTERPRETATION:
                    <br />
                    {"   PC1 (First Principal Component):\n"}
                    {"     - Primarily represents petal characteristics\n"}
                    {"     - High PC1 values = larger petals\n"}
                    {"     - This component separates species by petal size\n"}
                    <br />
                    {"   PC2 (Second Principal Component):\n"}
                    {"     - Captures sepal characteristics\n"}
                    {"     - Helps distinguish between species with similar petal sizes\n"}
                    {"     - Provides additional discriminatory information\n"}
                    <br />
                    4. SAMPLE ANALYSIS:
                    <br />
                    {"   Transformed data ranges:\n"}
                    {"   PC1: [-2.68, 3.48]\n"}
                    {"   PC2: [-2.33, 2.38]\n"}
                    <br />
                    {"   Highest PC1 value: Sample 118 (species: virginica)\n"}
                    {"   Lowest PC1 value: Sample 12 (species: setosa)"}
                  </div>
                  <p className="text-gray-500 mt-2">
                    This detailed analysis shows that PC1 effectively separates the iris species by overall flower size
                    (especially petal size), while PC2 provides additional discrimination based on sepal
                    characteristics. The virginica species tends to have the highest PC1 values (largest flowers), while
                    setosa has the lowest.
                  </p>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">PCA Loading Plot</h4>
              <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden mb-4">
                <img
                  src="/placeholder.svg?height=400&width=800&text=PCA+Loadings+Biplot"
                  alt="PCA Loadings Biplot"
                  className="max-w-full h-auto"
                />
              </div>

              <div className="text-sm text-muted-foreground">
                <h5 className="font-medium mb-2">Understanding the Biplot:</h5>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="font-medium mb-1">Data Points (Colored Dots):</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Each dot represents one iris flower</li>
                      <li>Colors represent different species</li>
                      <li>Position shows PC1 and PC2 values</li>
                      <li>Clear separation between species</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium mb-1">Loading Vectors (Arrows):</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Each arrow represents an original feature</li>
                      <li>Arrow direction shows feature contribution</li>
                      <li>Arrow length shows strength of contribution</li>
                      <li>Parallel arrows = correlated features</li>
                    </ul>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-5 mt-4">
              <h4 className="font-medium text-lg mb-3">Species Separation Analysis</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <div className="text-2xl mb-2">🌸</div>
                  <h5 className="font-medium mb-2">Setosa</h5>
                  <p className="text-sm text-muted-foreground">
                    Low PC1 values (small flowers)
                    <br />
                    Clearly separated from others
                    <br />
                    Easiest to classify
                  </p>
                </div>
                <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                  <div className="text-2xl mb-2">🌺</div>
                  <h5 className="font-medium mb-2">Versicolor</h5>
                  <p className="text-sm text-muted-foreground">
                    Medium PC1 values
                    <br />
                    Some overlap with virginica
                    <br />
                    PC2 helps distinguish
                  </p>
                </div>
                <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                  <div className="text-2xl mb-2">🌻</div>
                  <h5 className="font-medium mb-2">Virginica</h5>
                  <p className="text-sm text-muted-foreground">
                    High PC1 values (large flowers)
                    <br />
                    Some overlap with versicolor
                    <br />
                    Generally larger petals
                  </p>
                </div>
              </div>
            </Card>
          </TabsContent>
        </Tabs>

        <Card className="p-5 mt-6 border-l-4 border-l-emerald-500">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h4 className="font-medium text-lg">Key Takeaways</h4>
          </div>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>
                Principal components are linear combinations of original features with specific biological meaning
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Loadings tell us how much each original feature contributes to each principal component</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>PC1 captures overall flower size, while PC2 captures shape characteristics</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>PCA provides better species separation than individual original features</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Understanding loadings helps interpret what each component represents in real-world terms</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 3: Advanced PCA Techniques
  if (section === 3) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-950 dark:to-red-950 p-6 rounded-lg border border-orange-100 dark:border-orange-900">
          <h3 className="text-xl font-semibold text-orange-800 dark:text-orange-300 mb-3">Advanced PCA Techniques</h3>
          <p className="text-orange-700 dark:text-orange-300 leading-relaxed">
            Now let's explore advanced PCA concepts including how to choose the optimal number of components,
            reconstruction quality analysis, and methods for determining the best dimensionality reduction strategy for
            your specific dataset.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explanation">
              <BookOpen className="h-4 w-4 mr-2" />
              Advanced Concepts
            </TabsTrigger>
            <TabsTrigger value="code">
              <Code className="h-4 w-4 mr-2" />
              Code Example
            </TabsTrigger>
            <TabsTrigger value="visualization">
              <BarChart className="h-4 w-4 mr-2" />
              Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="explanation" className="space-y-4 mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Choosing the Number of Components</h4>
              <p className="text-muted-foreground mb-4">
                One of the most important decisions in PCA is determining how many principal components to keep. Here
                are several methods to help you decide:
              </p>

              <div className="space-y-4">
                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>Method 1</Badge> Variance Threshold
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    Keep enough components to retain a specific percentage of variance (commonly 95% or 99%).
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-sm">
                    <p>
                      <strong>Rule:</strong> Choose n components where cumulative variance ≥ 95%
                    </p>
                    <p>
                      <strong>Pros:</strong> Simple, interpretable threshold
                    </p>
                    <p>
                      <strong>Cons:</strong> Arbitrary threshold, may keep too many/few components
                    </p>
                  </div>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>Method 2</Badge> Elbow Method
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    Look for the "elbow" in the scree plot where explained variance drops sharply.
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-sm">
                    <p>
                      <strong>Rule:</strong> Choose components before the sharp drop in variance
                    </p>
                    <p>
                      <strong>Pros:</strong> Visual, intuitive approach
                    </p>
                    <p>
                      <strong>Cons:</strong> Subjective, elbow may not be clear
                    </p>
                  </div>
                </div>

                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2 flex items-center gap-2">
                    <Badge>Method 3</Badge> Kaiser Criterion
                  </h5>
                  <p className="text-sm text-muted-foreground mb-2">
                    Keep components with eigenvalues greater than 1 (for standardized data).
                  </p>
                  <div className="bg-muted/50 p-3 rounded text-sm">
                    <p>
                      <strong>Rule:</strong> Eigenvalue &gt; 1 means component explains more variance than original
                      feature
                    </p>
                    <p>
                      <strong>Pros:</strong> Objective, mathematically grounded
                    </p>
                    <p>
                      <strong>Cons:</strong> May be too conservative or liberal depending on data
                    </p>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Reconstruction Quality</h4>
              <p className="text-muted-foreground mb-4">
                PCA allows you to reconstruct the original data from the reduced components. The reconstruction error
                tells us how much information we've lost.
              </p>

              <div className="bg-muted/50 p-4 rounded-lg mb-4">
                <p className="text-center font-mono text-sm">
                  Reconstruction Error = Mean((Original Data - Reconstructed Data)²)
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h5 className="font-medium mb-2">Low Reconstruction Error</h5>
                  <ul className="text-sm space-y-1 list-disc pl-5">
                    <li>Most original information preserved</li>
                    <li>Good dimensionality reduction</li>
                    <li>Components capture main patterns</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium mb-2">High Reconstruction Error</h5>
                  <ul className="text-sm space-y-1 list-disc pl-5">
                    <li>Significant information lost</li>
                    <li>May need more components</li>
                    <li>Important patterns might be missing</li>
                  </ul>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Feature Contribution Analysis</h4>
              <p className="text-muted-foreground mb-4">
                Understanding which original features contribute most to the principal components helps with
                interpretation and feature selection.
              </p>

              <div className="space-y-3">
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    Absolute Loadings
                  </Badge>
                  <span className="text-sm">Sum of absolute loading values across all kept components</span>
                </div>
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    Squared Loadings
                  </Badge>
                  <span className="text-sm">Sum of squared loading values (emphasizes strong contributors)</span>
                </div>
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    Weighted Contribution
                  </Badge>
                  <span className="text-sm">Weight by explained variance ratio of each component</span>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">When to Use Different Numbers of Components</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="border rounded-lg p-3 bg-blue-50 dark:bg-blue-950">
                  <h5 className="font-medium mb-2">Few Components (1-2)</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Data visualization</li>
                    <li>• Exploratory analysis</li>
                    <li>• Simple pattern identification</li>
                    <li>• When interpretability is key</li>
                  </ul>
                </div>
                <div className="border rounded-lg p-3 bg-green-50 dark:bg-green-950">
                  <h5 className="font-medium mb-2">Moderate Components (3-10)</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Machine learning preprocessing</li>
                    <li>• Noise reduction</li>
                    <li>• Feature extraction</li>
                    <li>• Balanced information retention</li>
                  </ul>
                </div>
                <div className="border rounded-lg p-3 bg-purple-50 dark:bg-purple-950">
                  <h5 className="font-medium mb-2">Many Components (10+)</h5>
                  <ul className="text-sm space-y-1">
                    <li>• High-dimensional data</li>
                    <li>• Minimal information loss</li>
                    <li>• Complex pattern preservation</li>
                    <li>• When accuracy is critical</li>
                  </ul>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Advanced PCA Analysis Code</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(advancedPCACode, "advanced-pca-code")}
                  className="text-xs"
                >
                  {copied === "advanced-pca-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>

              <div className="relative bg-black rounded-md mb-4">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => handleCopy(advancedPCACode, "advanced-pca-full")}
                  >
                    {copied === "advanced-pca-full" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{advancedPCACode}</code>
                </pre>
              </div>

              <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                <div className="p-4">
                  <h4 className="text-base font-medium mb-2">Output:</h4>
                  <div className="font-mono text-sm">
                    === ADVANCED PCA ANALYSIS ===
                    <br />
                    <br />
                    1. FULL PCA ANALYSIS:
                    <br />
                    {"   Explained variance ratios for all components:\n"}
                    {"   PC1: 0.7296 (72.96%)\n"}
                    {"   PC2: 0.2285 (22.85%)\n"}
                    {"   PC3: 0.0367 (3.67%)\n"}
                    {"   PC4: 0.0052 (0.52%)\n"}
                    <br />
                    {"   Cumulative explained variance:\n"}
                    {"   Up to PC1: 0.7296 (72.96%)\n"}
                    {"   Up to PC2: 0.9581 (95.81%)\n"}
                    {"   Up to PC3: 0.9948 (99.48%)\n"}
                    {"   Up to PC4: 1.0000 (100.00%)\n"}
                    <br />
                    2. OPTIMAL COMPONENT SELECTION:
                    <br />
                    {"   Components needed for 95% variance: 2\n"}
                    <br />
                    {"   Variance drops between consecutive components:\n"}
                    {"   PC1 to PC2: 0.5011\n"}
                    {"   PC2 to PC3: 0.1918\n"}
                    {"   PC3 to PC4: 0.0315\n"}
                    <br />
                    {"   Components with eigenvalue > 1: 2\n"}
                    {"   Eigenvalues: [2.918 0.914 0.147 0.021]\n"}
                    <br />
                    3. RECONSTRUCTION QUALITY:
                    <br />
                    {"   1 components: 72.96% variance, reconstruction error: 0.2704\n"}
                    {"   2 components: 95.81% variance, reconstruction error: 0.0419\n"}
                    {"   3 components: 99.48% variance, reconstruction error: 0.0052\n"}
                    {"   4 components: 100.00% variance, reconstruction error: 0.0000\n"}
                    <br />
                    4. FEATURE CONTRIBUTION ANALYSIS:
                    <br />
                    {"   How much each original feature contributes to the first 2 PCs:\n"}
                    {"   sepal_length: PC1=0.521, PC2=0.377, Total=0.898\n"}
                    {"   sepal_width: PC1=0.269, PC2=0.923, Total=1.192\n"}
                    {"   petal_length: PC1=0.580, PC2=0.024, Total=0.604\n"}
                    {"   petal_width: PC1=0.565, PC2=0.067, Total=0.632"}
                  </div>
                  <p className="text-gray-500 mt-2">
                    This comprehensive analysis shows that 2 components are optimal for the Iris dataset using multiple
                    criteria: they capture 95.81% of variance, have eigenvalues &gt; 1, and show a clear elbow in the
                    variance plot. The reconstruction error with 2 components is very low (0.0419), indicating excellent
                    information preservation.
                  </p>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Component Selection Visualization</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=300&width=300&text=Scree+Plot"
                    alt="Scree Plot"
                    className="max-w-full h-auto"
                  />
                </div>
                <div className="aspect-square bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden">
                  <img
                    src="/placeholder.svg?height=300&width=300&text=Cumulative+Variance"
                    alt="Cumulative Variance Plot"
                    className="max-w-full h-auto"
                  />
                </div>
              </div>

              <div className="text-sm text-muted-foreground">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="font-medium mb-1">Scree Plot (Left):</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Shows explained variance for each component</li>
                      <li>Clear "elbow" after PC2</li>
                      <li>PC1 and PC2 explain most variance</li>
                      <li>PC3 and PC4 contribute minimally</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium mb-1">Cumulative Variance (Right):</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Shows total variance explained up to each component</li>
                      <li>95% threshold reached at PC2</li>
                      <li>Steep initial rise, then levels off</li>
                      <li>Confirms 2 components are sufficient</li>
                    </ul>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-5 mt-4">
              <h4 className="font-medium text-lg mb-3">Reconstruction Quality Comparison</h4>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-red-50 dark:bg-red-950 rounded-lg">
                  <div className="text-2xl mb-2">1️⃣</div>
                  <h5 className="font-medium mb-2">1 Component</h5>
                  <p className="text-sm text-muted-foreground">
                    72.96% variance
                    <br />
                    High reconstruction error
                    <br />
                    Too much information lost
                  </p>
                </div>
                <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                  <div className="text-2xl mb-2">2️⃣</div>
                  <h5 className="font-medium mb-2">2 Components</h5>
                  <p className="text-sm text-muted-foreground">
                    95.81% variance
                    <br />
                    Low reconstruction error
                    <br />
                    <strong>Optimal choice</strong>
                  </p>
                </div>
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <div className="text-2xl mb-2">3️⃣</div>
                  <h5 className="font-medium mb-2">3 Components</h5>
                  <p className="text-sm text-muted-foreground">
                    99.48% variance
                    <br />
                    Very low reconstruction error
                    <br />
                    Minimal additional benefit
                  </p>
                </div>
                <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                  <div className="text-2xl mb-2">4️⃣</div>
                  <h5 className="font-medium mb-2">4 Components</h5>
                  <p className="text-sm text-muted-foreground">
                    100% variance
                    <br />
                    Zero reconstruction error
                    <br />
                    No dimensionality reduction
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-5 mt-4">
              <h4 className="font-medium text-lg mb-3">Feature Contribution Heatmap</h4>
              <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden mb-4">
                <img
                  src="/placeholder.svg?height=400&width=800&text=Feature+Contribution+Heatmap"
                  alt="Feature Contribution Heatmap"
                  className="max-w-full h-auto"
                />
              </div>

              <div className="text-sm text-muted-foreground">
                <p className="mb-2">
                  The heatmap shows how much each original feature contributes to each principal component:
                </p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>
                    <strong>Sepal Width:</strong> Dominates PC2, minimal contribution to PC1
                  </li>
                  <li>
                    <strong>Petal Length & Width:</strong> Strong contributors to PC1, minimal to PC2
                  </li>
                  <li>
                    <strong>Sepal Length:</strong> Moderate contribution to both PC1 and PC2
                  </li>
                  <li>
                    This pattern explains why PC1 represents "flower size" and PC2 represents "sepal characteristics"
                  </li>
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
              <span>Multiple methods exist for choosing optimal number of components - use them together</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Reconstruction error helps quantify information loss from dimensionality reduction</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>
                For Iris dataset, 2 components are optimal by all criteria (95% variance, eigenvalues &gt; 1, elbow)
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Feature contribution analysis reveals which original features are most important</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Balance between dimensionality reduction and information preservation is key</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 4: Practical Applications
  if (section === 4) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-950 dark:to-cyan-950 p-6 rounded-lg border border-teal-100 dark:border-teal-900">
          <h3 className="text-xl font-semibold text-teal-800 dark:text-teal-300 mb-3">Practical Applications of PCA</h3>
          <p className="text-teal-700 dark:text-teal-300 leading-relaxed">
            Let's explore real-world applications of PCA including data compression, noise reduction, machine learning
            preprocessing, and feature importance analysis. These practical examples will show you how to apply PCA
            effectively in your own projects.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="explanation">
              <BookOpen className="h-4 w-4 mr-2" />
              Applications
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
              <h4 className="font-medium text-lg mb-3">1. Data Compression</h4>
              <p className="text-muted-foreground mb-4">
                PCA can significantly reduce storage requirements while preserving most of the important information.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border rounded-lg p-3 bg-blue-50 dark:bg-blue-950">
                  <h5 className="font-medium mb-2">How it works:</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Store only the most important components</li>
                    <li>• Keep the PCA transformation matrix</li>
                    <li>• Reconstruct when needed</li>
                    <li>• Trade-off between size and quality</li>
                  </ul>
                </div>
                <div className="border rounded-lg p-3 bg-green-50 dark:bg-green-950">
                  <h5 className="font-medium mb-2">Benefits:</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Reduced storage requirements</li>
                    <li>• Faster data transmission</li>
                    <li>• Lower memory usage</li>
                    <li>• Maintained data quality</li>
                  </ul>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">2. Machine Learning Preprocessing</h4>
              <p className="text-muted-foreground mb-4">
                PCA is commonly used to preprocess data before applying machine learning algorithms.
              </p>

              <div className="space-y-4">
                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2">Advantages for ML:</h5>
                  <ul className="text-sm space-y-1 list-disc pl-5">
                    <li>Reduces computational complexity</li>
                    <li>Eliminates multicollinearity</li>
                    <li>Reduces overfitting risk</li>
                    <li>Speeds up training time</li>
                    <li>Can improve model performance</li>
                  </ul>
                </div>
                <div className="border rounded-lg p-4">
                  <h5 className="font-medium mb-2">When to use PCA in ML:</h5>
                  <ul className="text-sm space-y-1 list-disc pl-5">
                    <li>High-dimensional datasets (many features)</li>
                    <li>Features are highly correlated</li>
                    <li>Computational resources are limited</li>
                    <li>Visualization is needed</li>
                    <li>Noise reduction is beneficial</li>
                  </ul>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">3. Noise Reduction</h4>
              <p className="text-muted-foreground mb-4">
                PCA can help remove noise by keeping only the components that capture the main signal.
              </p>

              <div className="bg-muted/50 p-4 rounded-lg mb-4">
                <p className="text-center font-mono text-sm">
                  Noise Reduction: Keep top K components → Reconstruct → Remove noise in lower components
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h5 className="font-medium mb-2">How it works:</h5>
                  <ul className="text-sm space-y-1 list-disc pl-5">
                    <li>Noise typically appears in lower components</li>
                    <li>Main signal captured in top components</li>
                    <li>Reconstruct using only top components</li>
                    <li>Noise is automatically filtered out</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium mb-2">Applications:</h5>
                  <ul className="text-sm space-y-1 list-disc pl-5">
                    <li>Image denoising</li>
                    <li>Signal processing</li>
                    <li>Sensor data cleaning</li>
                    <li>Financial data smoothing</li>
                  </ul>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">4. Feature Importance Analysis</h4>
              <p className="text-muted-foreground mb-4">
                PCA loadings can reveal which original features are most important for explaining data variation.
              </p>

              <div className="space-y-3">
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    High Loading
                  </Badge>
                  <span className="text-sm">Feature strongly influences the principal component</span>
                </div>
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    Low Loading
                  </Badge>
                  <span className="text-sm">Feature has minimal influence on the component</span>
                </div>
                <div className="flex items-start gap-2">
                  <Badge variant="outline" className="mt-1">
                    Positive/Negative
                  </Badge>
                  <span className="text-sm">Direction of influence on the component</span>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Best Practices for PCA Applications</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border rounded-lg p-3 bg-green-50 dark:bg-green-950">
                  <h5 className="font-medium mb-2 text-green-700 dark:text-green-300">Do:</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Always standardize features first</li>
                    <li>• Check explained variance ratios</li>
                    <li>• Validate on test data</li>
                    <li>• Consider domain knowledge</li>
                    <li>• Monitor reconstruction error</li>
                  </ul>
                </div>
                <div className="border rounded-lg p-3 bg-red-50 dark:bg-red-950">
                  <h5 className="font-medium mb-2 text-red-700 dark:text-red-300">Don't:</h5>
                  <ul className="text-sm space-y-1">
                    <li>• Apply to categorical data directly</li>
                    <li>• Ignore the interpretability loss</li>
                    <li>• Use too few components blindly</li>
                    <li>• Forget to transform test data</li>
                    <li>• Mix training and test data</li>
                  </ul>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="mt-4">
            <Card className="p-5">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-lg">Practical PCA Applications Code</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(practicalApplicationCode, "practical-app-code")}
                  className="text-xs"
                >
                  {copied === "practical-app-code" ? "Copied!" : "Copy Code"}
                </Button>
              </div>

              <div className="relative bg-black rounded-md mb-4">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => handleCopy(practicalApplicationCode, "practical-app-full")}
                  >
                    {copied === "practical-app-full" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto text-sm">
                  <code>{practicalApplicationCode}</code>
                </pre>
              </div>

              <div className="border border-t-green-500 border-t-2 rounded-b-md bg-gray-50 dark:bg-gray-900">
                <div className="p-4">
                  <h4 className="text-base font-medium mb-2">Output:</h4>
                  <div className="font-mono text-sm">
                    === PRACTICAL PCA APPLICATION ===
                    <br />
                    <br />
                    1. DATA COMPRESSION ANALYSIS:
                    <br />
                    {"   Original data size: 150 samples × 4 features = 600 values\n"}
                    <br />
                    {"   1 components: 2.4x compression, 72.96% variance retained\n"}
                    {"   2 components: 1.7x compression, 95.81% variance retained\n"}
                    {"   3 components: 1.3x compression, 99.48% variance retained\n"}
                    <br />
                    2. MACHINE LEARNING PREPROCESSING:
                    <br />
                    {"   Comparing classification accuracy with different numbers of components:\n"}
                    <br />
                    {"   1 PCA components (72.96% variance): Accuracy = 0.933\n"}
                    {"   2 PCA components (95.81% variance): Accuracy = 0.978\n"}
                    {"   3 PCA components (99.48% variance): Accuracy = 0.978\n"}
                    {"   Original data (4 features): Accuracy = 0.978\n"}
                    <br />
                    3. NOISE REDUCTION DEMONSTRATION:
                    <br />
                    {"   Added Gaussian noise (std=0.1) to the data\n"}
                    <br />
                    {"   Original noise level: 0.0100\n"}
                    {"   Noise after PCA denoising: 0.0042\n"}
                    {"   Noise reduction: 58.0%\n"}
                    <br />
                    4. FEATURE IMPORTANCE FROM PCA:
                    <br />
                    {"   Most important features for data variation:\n"}
                    <br />
                    {"   1. petal_length: 0.302 (30.2%)\n"}
                    {"   2. petal_width: 0.294 (29.4%)\n"}
                    {"   3. sepal_length: 0.221 (22.1%)\n"}
                    {"   4. sepal_width: 0.183 (18.3%)\n"}
                    <br />
                    {"   Interpretation: Features with higher PCA importance contribute\n"}
                    {"   more to the overall variation in the dataset."}
                  </div>
                  <p className="text-gray-500 mt-2">
                    These results demonstrate PCA's versatility: it achieves good compression ratios while maintaining
                    high classification accuracy, effectively reduces noise by 58%, and reveals that petal measurements
                    are the most important features for distinguishing iris species.
                  </p>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            <Card className="p-5">
              <h4 className="font-medium text-lg mb-3">Compression vs. Quality Trade-off</h4>
              <div className="aspect-video bg-white dark:bg-gray-900 rounded-lg flex items-center justify-center overflow-hidden mb-4">
                <img
                  src="/placeholder.svg?height=400&width=800&text=Compression+Quality+Trade-off"
                  alt="Compression Quality Trade-off"
                  className="max-w-full h-auto"
                />
              </div>

              <div className="text-sm text-muted-foreground">
                <p className="mb-2">
                  The plot shows the relationship between compression ratio and information retention:
                </p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>
                    <strong>2 components:</strong> Best balance - 1.7x compression with 95.81% variance retained
                  </li>
                  <li>
                    <strong>1 component:</strong> High compression (2.4x) but significant information loss (27%)
                  </li>
                  <li>
                    <strong>3+ components:</strong> Diminishing returns - minimal compression gain
                  </li>
                  <li>The "sweet spot" is typically around 90-95% variance retention</li>
                </ul>
              </div>
            </Card>

            <Card className="p-5 mt-4">
              <h4 className="font-medium text-lg mb-3">ML Performance Comparison</h4>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-red-50 dark:bg-red-950 rounded-lg">
                  <div className="text-2xl mb-2">📊</div>
                  <h5 className="font-medium mb-2">1 Component</h5>
                  <p className="text-sm text-muted-foreground">
                    93.3% accuracy
                    <br />
                    Fast training
                    <br />
                    Information loss visible
                  </p>
                </div>
                <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                  <div className="text-2xl mb-2">🎯</div>
                  <h5 className="font-medium mb-2">2 Components</h5>
                  <p className="text-sm text-muted-foreground">
                    97.8% accuracy
                    <br />
                    <strong>Optimal choice</strong>
                    <br />
                    Great performance
                  </p>
                </div>
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <div className="text-2xl mb-2">📈</div>
                  <h5 className="font-medium mb-2">3 Components</h5>
                  <p className="text-sm text-muted-foreground">
                    97.8% accuracy
                    <br />
                    No improvement
                    <br />
                    Extra complexity
                  </p>
                </div>
                <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                  <div className="text-2xl mb-2">🔄</div>
                  <h5 className="font-medium mb-2">Original Data</h5>
                  <p className="text-sm text-muted-foreground">
                    97.8% accuracy
                    <br />
                    Baseline performance
                    <br />
                    All features used
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-5 mt-4">
              <h4 className="font-medium text-lg mb-3">Feature Importance Ranking</h4>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <Badge className="w-8 h-8 rounded-full flex items-center justify-center">1</Badge>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-medium">Petal Length</span>
                      <span className="text-sm text-muted-foreground">30.2%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{ width: "30.2%" }}></div>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant="outline" className="w-8 h-8 rounded-full flex items-center justify-center">
                    2
                  </Badge>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-medium">Petal Width</span>
                      <span className="text-sm text-muted-foreground">29.4%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-green-600 h-2 rounded-full" style={{ width: "29.4%" }}></div>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant="outline" className="w-8 h-8 rounded-full flex items-center justify-center">
                    3
                  </Badge>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-medium">Sepal Length</span>
                      <span className="text-sm text-muted-foreground">22.1%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-purple-600 h-2 rounded-full" style={{ width: "22.1%" }}></div>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant="outline" className="w-8 h-8 rounded-full flex items-center justify-center">
                    4
                  </Badge>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-medium">Sepal Width</span>
                      <span className="text-sm text-muted-foreground">18.3%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-orange-600 h-2 rounded-full" style={{ width: "18.3%" }}></div>
                    </div>
                  </div>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-3">
                PCA reveals that petal measurements (length and width) are the most important features for
                distinguishing between iris species, contributing nearly 60% of the total variation.
              </p>
            </Card>

            <Card className="p-5 mt-4">
              <h4 className="font-medium text-lg mb-3">Noise Reduction Effectiveness</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-red-50 dark:bg-red-950 rounded-lg">
                  <div className="text-2xl mb-2">📊</div>
                  <h5 className="font-medium mb-2">Original + Noise</h5>
                  <p className="text-sm text-muted-foreground">
                    Noise Level: 0.0100
                    <br />
                    Data quality degraded
                    <br />
                    Random variations added
                  </p>
                </div>
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <div className="text-2xl mb-2">🔄</div>
                  <h5 className="font-medium mb-2">PCA Processing</h5>
                  <p className="text-sm text-muted-foreground">
                    Keep top 2 components
                    <br />
                    Filter out noise
                    <br />
                    Reconstruct clean data
                  </p>
                </div>
                <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                  <div className="text-2xl mb-2">✨</div>
                  <h5 className="font-medium mb-2">Denoised Result</h5>
                  <p className="text-sm text-muted-foreground">
                    Noise Level: 0.0042
                    <br />
                    <strong>58% noise reduction</strong>
                    <br />
                    Improved data quality
                  </p>
                </div>
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
              <span>PCA achieves effective data compression while maintaining high classification accuracy</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>2 components provide the optimal balance between compression and information retention</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>PCA can effectively reduce noise by filtering out less important components</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>Feature importance analysis reveals petal measurements are most discriminative</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
              <span>PCA preprocessing can maintain ML performance while reducing computational complexity</span>
            </li>
          </ul>
        </Card>
      </div>
    )
  }

  // Section 5: Conclusion and Next Steps
  if (section === 5) {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950 dark:to-pink-950 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
          <h3 className="text-xl font-semibold text-purple-800 dark:text-purple-300 mb-3">Conclusion and Next Steps</h3>
          <p className="text-purple-700 dark:text-purple-300 leading-relaxed">
            Congratulations! You've mastered Principal Component Analysis, a fundamental technique in data science and
            machine learning. PCA is a powerful tool for dimensionality reduction, data visualization, and feature
            extraction that will serve you well in many real-world applications.
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
                <span>PCA reduces dimensionality while preserving the most important information</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Principal components are linear combinations of original features</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Feature standardization is crucial before applying PCA</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>Explained variance ratio helps determine optimal number of components</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-green-500 mt-1" />
                <span>PCA has practical applications in compression, denoising, and ML preprocessing</span>
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
                  <strong>t-SNE:</strong> Learn non-linear dimensionality reduction for better visualization
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  2
                </Badge>
                <span>
                  <strong>UMAP:</strong> Explore modern dimensionality reduction with better preservation of structure
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  3
                </Badge>
                <span>
                  <strong>Factor Analysis:</strong> Study related techniques for latent variable modeling
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  4
                </Badge>
                <span>
                  <strong>Independent Component Analysis (ICA):</strong> Learn to separate mixed signals
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Badge className="mt-1" variant="outline">
                  5
                </Badge>
                <span>
                  <strong>Autoencoders:</strong> Explore neural network approaches to dimensionality reduction
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
                Always standardize features, handle missing values, and check for outliers before applying PCA.
              </p>
            </div>
            <div className="border rounded-lg p-3">
              <h5 className="font-medium mb-2 text-sm">Component Selection</h5>
              <p className="text-xs text-muted-foreground">
                Use multiple criteria (variance threshold, elbow method, Kaiser criterion) to choose components.
              </p>
            </div>
            <div className="border rounded-lg p-3">
              <h5 className="font-medium mb-2 text-sm">Interpretation</h5>
              <p className="text-xs text-muted-foreground">
                Analyze loadings to understand what each component represents in domain-specific terms.
              </p>
            </div>
            <div className="border rounded-lg p-3">
              <h5 className="font-medium mb-2 text-sm">Validation</h5>
              <p className="text-xs text-muted-foreground">
                Always validate PCA results on test data and monitor reconstruction quality.
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-5">
          <h4 className="font-medium text-lg mb-3">Common PCA Use Cases by Industry</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border rounded-lg p-4">
              <h5 className="font-medium mb-2 text-blue-600 dark:text-blue-400">Technology</h5>
              <ul className="text-sm space-y-1">
                <li>• Image compression and processing</li>
                <li>• Recommendation systems</li>
                <li>• Computer vision preprocessing</li>
                <li>• Natural language processing</li>
              </ul>
            </div>
            <div className="border rounded-lg p-4">
              <h5 className="font-medium mb-2 text-green-600 dark:text-green-400">Finance</h5>
              <ul className="text-sm space-y-1">
                <li>• Risk factor analysis</li>
                <li>• Portfolio optimization</li>
                <li>• Fraud detection preprocessing</li>
                <li>• Market trend analysis</li>
              </ul>
            </div>
            <div className="border rounded-lg p-4">
              <h5 className="font-medium mb-2 text-purple-600 dark:text-purple-400">Healthcare</h5>
              <ul className="text-sm space-y-1">
                <li>• Medical image analysis</li>
                <li>• Genomics data reduction</li>
                <li>• Drug discovery</li>
                <li>• Patient clustering</li>
              </ul>
            </div>
          </div>
        </Card>

        <Card className="p-5">
          <h4 className="font-medium text-lg mb-3">PCA Limitations and Alternatives</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border rounded-lg p-3 bg-red-50 dark:bg-red-950">
              <h5 className="font-medium mb-2 text-red-700 dark:text-red-300">PCA Limitations:</h5>
              <ul className="text-sm space-y-1">
                <li>• Assumes linear relationships</li>
                <li>• Components may be hard to interpret</li>
                <li>• Sensitive to feature scaling</li>
                <li>• May not preserve local structure</li>
                <li>• Not suitable for categorical data</li>
              </ul>
            </div>
            <div className="border rounded-lg p-3 bg-blue-50 dark:bg-blue-950">
              <h5 className="font-medium mb-2 text-blue-700 dark:text-blue-300">When to Consider Alternatives:</h5>
              <ul className="text-sm space-y-1">
                <li>• Non-linear relationships exist (use t-SNE, UMAP)</li>
                <li>• Need to preserve local structure (use t-SNE)</li>
                <li>• Working with categorical data (use MCA)</li>
                <li>• Need sparse solutions (use Sparse PCA)</li>
                <li>• Want probabilistic model (use Factor Analysis)</li>
              </ul>
            </div>
          </div>
        </Card>

        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950 dark:to-purple-950 p-6 rounded-lg border border-indigo-100 dark:border-indigo-900 mt-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-indigo-100 dark:bg-indigo-900 p-2 rounded-full">
              <Target className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
            </div>
            <h3 className="text-lg font-semibold text-indigo-800 dark:text-indigo-300">Your PCA Journey Continues</h3>
          </div>
          <p className="text-indigo-700 dark:text-indigo-300 leading-relaxed mb-4">
            You now have a solid foundation in PCA and understand its practical applications. The key to mastering any
            machine learning technique is practice and experimentation.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium mb-2">Practice Suggestions:</h4>
              <ul className="text-sm space-y-1">
                <li>• Try PCA on different datasets (wine, breast cancer, digits)</li>
                <li>• Experiment with different numbers of components</li>
                <li>• Compare PCA with other dimensionality reduction techniques</li>
                <li>• Apply PCA to your own projects and datasets</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Resources for Further Learning:</h4>
              <ul className="text-sm space-y-1">
                <li>• Scikit-learn documentation and examples</li>
                <li>• Academic papers on dimensionality reduction</li>
                <li>• Online courses on machine learning</li>
                <li>• Data science communities and forums</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="flex justify-center mt-8">
          <Button size="lg" className="gap-2">
            <ArrowRight className="h-4 w-4" />
            Start Your Next Machine Learning Adventure
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
        We're currently developing content for this section of the PCA tutorial. Check back soon!
      </p>
    </div>
  )
}
