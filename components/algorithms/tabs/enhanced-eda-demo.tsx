"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { RefreshCw, BarChart3, ScatterChart, Table2, Target, Sigma } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Colors,
  Filler,
  ChartData,
} from "chart.js"
import { Bar, Scatter } from "react-chartjs-2"

// Register Chart.js components
Chart.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Colors,
  Filler,
)

// Real-world datasets
const DATASETS = {
  iris: {
    name: "Iris Flower Dataset",
    description: "Famous dataset containing measurements of iris flowers with 3 different species",
    features: ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    target: "species",
    targetValues: ["setosa", "versicolor", "virginica"],
    size: 150,
    data: [
      // Sample of the Iris dataset (first 10 rows of each class)
      { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2, species: "setosa" },
      { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2, species: "setosa" },
      { sepal_length: 4.7, sepal_width: 3.2, petal_length: 1.3, petal_width: 0.2, species: "setosa" },
      { sepal_length: 4.6, sepal_width: 3.1, petal_length: 1.5, petal_width: 0.2, species: "setosa" },
      { sepal_length: 5.0, sepal_width: 3.6, petal_length: 1.4, petal_width: 0.2, species: "setosa" },
      { sepal_length: 5.4, sepal_width: 3.9, petal_length: 1.7, petal_width: 0.4, species: "setosa" },
      { sepal_length: 4.6, sepal_width: 3.4, petal_length: 1.4, petal_width: 0.3, species: "setosa" },
      { sepal_length: 5.0, sepal_width: 3.4, petal_length: 1.5, petal_width: 0.2, species: "setosa" },
      { sepal_length: 4.4, sepal_width: 2.9, petal_length: 1.4, petal_width: 0.2, species: "setosa" },
      { sepal_length: 4.9, sepal_width: 3.1, petal_length: 1.5, petal_width: 0.1, species: "setosa" },
      { sepal_length: 7.0, sepal_width: 3.2, petal_length: 4.7, petal_width: 1.4, species: "versicolor" },
      { sepal_length: 6.4, sepal_width: 3.2, petal_length: 4.5, petal_width: 1.5, species: "versicolor" },
      { sepal_length: 6.9, sepal_width: 3.1, petal_length: 4.9, petal_width: 1.5, species: "versicolor" },
      { sepal_length: 5.5, sepal_width: 2.3, petal_length: 4.0, petal_width: 1.3, species: "versicolor" },
      { sepal_length: 6.5, sepal_width: 2.8, petal_length: 4.6, petal_width: 1.5, species: "versicolor" },
      { sepal_length: 5.7, sepal_width: 2.8, petal_length: 4.5, petal_width: 1.3, species: "versicolor" },
      { sepal_length: 6.3, sepal_width: 3.3, petal_length: 4.7, petal_width: 1.6, species: "versicolor" },
      { sepal_length: 4.9, sepal_width: 2.4, petal_length: 3.3, petal_width: 1.0, species: "versicolor" },
      { sepal_length: 6.6, sepal_width: 2.9, petal_length: 4.6, petal_width: 1.3, species: "versicolor" },
      { sepal_length: 5.2, sepal_width: 2.7, petal_length: 3.9, petal_width: 1.4, species: "versicolor" },
      { sepal_length: 6.3, sepal_width: 3.3, petal_length: 6.0, petal_width: 2.5, species: "virginica" },
      { sepal_length: 5.8, sepal_width: 2.7, petal_length: 5.1, petal_width: 1.9, species: "virginica" },
      { sepal_length: 7.1, sepal_width: 3.0, petal_length: 5.9, petal_width: 2.1, species: "virginica" },
      { sepal_length: 6.3, sepal_width: 2.9, petal_length: 5.6, petal_width: 1.8, species: "virginica" },
      { sepal_length: 6.5, sepal_width: 3.0, petal_length: 5.8, petal_width: 2.2, species: "virginica" },
      { sepal_length: 7.6, sepal_width: 3.0, petal_length: 6.6, petal_width: 2.1, species: "virginica" },
      { sepal_length: 4.9, sepal_width: 2.5, petal_length: 4.5, petal_width: 1.7, species: "virginica" },
      { sepal_length: 7.3, sepal_width: 2.9, petal_length: 6.3, petal_width: 1.8, species: "virginica" },
      { sepal_length: 6.7, sepal_width: 2.5, petal_length: 5.8, petal_width: 1.8, species: "virginica" },
      { sepal_length: 7.2, sepal_width: 3.6, petal_length: 6.1, petal_width: 2.5, species: "virginica" },
    ],
  },
  boston: {
    name: "Boston Housing Dataset",
    description: "Housing data for 506 census tracts in Boston with various features",
    features: ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"],
    target: "medv",
    size: 506,
    data: [
      // Sample of the Boston Housing dataset (first 15 rows)
      {
        crim: 0.00632,
        zn: 18.0,
        indus: 2.31,
        chas: 0,
        nox: 0.538,
        rm: 6.575,
        age: 65.2,
        dis: 4.09,
        rad: 1,
        tax: 296,
        ptratio: 15.3,
        b: 396.9,
        lstat: 4.98,
        medv: 24.0,
      },
      {
        crim: 0.02731,
        zn: 0.0,
        indus: 7.07,
        chas: 0,
        nox: 0.469,
        rm: 6.421,
        age: 78.9,
        dis: 4.9671,
        rad: 2,
        tax: 242,
        ptratio: 17.8,
        b: 396.9,
        lstat: 9.14,
        medv: 21.6,
      },
      {
        crim: 0.02729,
        zn: 0.0,
        indus: 7.07,
        chas: 0,
        nox: 0.469,
        rm: 7.185,
        age: 61.1,
        dis: 4.9671,
        rad: 2,
        tax: 242,
        ptratio: 17.8,
        b: 392.83,
        lstat: 4.03,
        medv: 34.7,
      },
      {
        crim: 0.03237,
        zn: 0.0,
        indus: 2.18,
        chas: 0,
        nox: 0.458,
        rm: 6.998,
        age: 45.8,
        dis: 6.0622,
        rad: 3,
        tax: 222,
        ptratio: 18.7,
        b: 394.63,
        lstat: 2.94,
        medv: 33.4,
      },
      {
        crim: 0.06905,
        zn: 0.0,
        indus: 2.18,
        chas: 0,
        nox: 0.458,
        rm: 7.147,
        age: 54.2,
        dis: 6.0622,
        rad: 3,
        tax: 222,
        ptratio: 18.7,
        b: 396.9,
        lstat: 5.33,
        medv: 36.2,
      },
      {
        crim: 0.02985,
        zn: 0.0,
        indus: 2.18,
        chas: 0,
        nox: 0.458,
        rm: 6.43,
        age: 58.7,
        dis: 6.0622,
        rad: 3,
        tax: 222,
        ptratio: 18.7,
        b: 394.12,
        lstat: 5.21,
        medv: 28.7,
      },
      {
        crim: 0.08829,
        zn: 12.5,
        indus: 7.87,
        chas: 0,
        nox: 0.524,
        rm: 6.012,
        age: 66.6,
        dis: 5.5605,
        rad: 5,
        tax: 311,
        ptratio: 15.2,
        b: 395.6,
        lstat: 12.43,
        medv: 22.9,
      },
      {
        crim: 0.14455,
        zn: 12.5,
        indus: 7.87,
        chas: 0,
        nox: 0.524,
        rm: 6.172,
        age: 96.1,
        dis: 5.9505,
        rad: 5,
        tax: 311,
        ptratio: 15.2,
        b: 396.9,
        lstat: 19.15,
        medv: 27.1,
      },
      {
        crim: 0.21124,
        zn: 12.5,
        indus: 7.87,
        chas: 0,
        nox: 0.524,
        rm: 5.631,
        age: 100.0,
        dis: 6.0821,
        rad: 5,
        tax: 311,
        ptratio: 15.2,
        b: 386.63,
        lstat: 29.93,
        medv: 16.5,
      },
      {
        crim: 0.17004,
        zn: 12.5,
        indus: 7.87,
        chas: 0,
        nox: 0.524,
        rm: 6.004,
        age: 85.9,
        dis: 6.5921,
        rad: 5,
        tax: 311,
        ptratio: 15.2,
        b: 386.71,
        lstat: 17.1,
        medv: 18.9,
      },
      {
        crim: 0.22489,
        zn: 12.5,
        indus: 7.87,
        chas: 0,
        nox: 0.524,
        rm: 6.377,
        age: 94.3,
        dis: 6.3467,
        rad: 5,
        tax: 311,
        ptratio: 15.2,
        b: 392.52,
        lstat: 20.45,
        medv: 15.0,
      },
      {
        crim: 0.11747,
        zn: 12.5,
        indus: 7.87,
        chas: 0,
        nox: 0.524,
        rm: 6.009,
        age: 82.9,
        dis: 6.2267,
        rad: 5,
        tax: 311,
        ptratio: 15.2,
        b: 396.9,
        lstat: 13.27,
        medv: 18.9,
      },
      {
        crim: 0.09378,
        zn: 12.5,
        indus: 7.87,
        chas: 0,
        nox: 0.524,
        rm: 5.889,
        age: 39.0,
        dis: 5.4509,
        rad: 5,
        tax: 311,
        ptratio: 15.2,
        b: 390.5,
        lstat: 15.71,
        medv: 21.7,
      },
      {
        crim: 0.62976,
        zn: 0.0,
        indus: 8.14,
        chas: 0,
        nox: 0.538,
        rm: 5.949,
        age: 61.8,
        dis: 4.7075,
        rad: 4,
        tax: 307,
        ptratio: 21.0,
        b: 396.9,
        lstat: 8.26,
        medv: 20.4,
      },
      {
        crim: 0.63796,
        zn: 0.0,
        indus: 8.14,
        chas: 0,
        nox: 0.538,
        rm: 6.096,
        age: 84.5,
        dis: 4.4619,
        rad: 4,
        tax: 307,
        ptratio: 21.0,
        b: 380.02,
        lstat: 10.26,
        medv: 18.2,
      },
    ],
  },
  diabetes: {
    name: "Diabetes Dataset",
    description: "Diabetes patient records with various health metrics and disease progression",
    features: ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
    target: "progression",
    size: 442,
    data: [
      // Sample of the Diabetes dataset (first 15 rows)
      { age: 59, sex: 2, bmi: 32.1, bp: 101.0, s1: 157, s2: 93.2, s3: 38, s4: 4, s5: 4.9, s6: 87, progression: 151 },
      { age: 48, sex: 1, bmi: 21.6, bp: 87.0, s1: 183, s2: 103.2, s3: 70, s4: 3, s5: 3.9, s6: 69, progression: 75 },
      { age: 72, sex: 2, bmi: 30.5, bp: 93.0, s1: 156, s2: 93.6, s3: 41, s4: 4, s5: 4.7, s6: 85, progression: 141 },
      { age: 24, sex: 1, bmi: 25.3, bp: 84.0, s1: 198, s2: 131.4, s3: 40, s4: 5, s5: 4.9, s6: 89, progression: 206 },
      { age: 50, sex: 1, bmi: 23.0, bp: 101.0, s1: 192, s2: 125.4, s3: 52, s4: 4, s5: 4.3, s6: 80, progression: 135 },
      { age: 23, sex: 1, bmi: 22.6, bp: 89.0, s1: 139, s2: 64.8, s3: 61, s4: 2, s5: 4.2, s6: 68, progression: 97 },
      { age: 36, sex: 2, bmi: 22.0, bp: 90.0, s1: 160, s2: 99.6, s3: 50, s4: 3, s5: 3.9, s6: 82, progression: 138 },
      { age: 66, sex: 2, bmi: 26.2, bp: 114.0, s1: 255, s2: 185.0, s3: 56, s4: 4, s5: 4.3, s6: 92, progression: 152 },
      { age: 60, sex: 2, bmi: 32.1, bp: 83.0, s1: 179, s2: 119.4, s3: 42, s4: 4, s5: 4.9, s6: 94, progression: 220 },
      { age: 29, sex: 1, bmi: 30.0, bp: 85.0, s1: 180, s2: 93.4, s3: 43, s4: 4, s5: 5.1, s6: 101, progression: 171 },
      { age: 22, sex: 1, bmi: 18.6, bp: 97.0, s1: 114, s2: 57.6, s3: 46, s4: 2, s5: 3.9, s6: 83, progression: 86 },
      { age: 56, sex: 2, bmi: 28.0, bp: 85.0, s1: 184, s2: 144.0, s3: 32, s4: 6, s5: 4.1, s6: 89, progression: 175 },
      { age: 53, sex: 1, bmi: 23.7, bp: 92.0, s1: 186, s2: 109.0, s3: 62, s4: 3, s5: 4.2, s6: 90, progression: 134 },
      { age: 50, sex: 2, bmi: 26.2, bp: 97.0, s1: 186, s2: 105.4, s3: 49, s4: 4, s5: 5.1, s6: 96, progression: 194 },
      { age: 46, sex: 1, bmi: 27.2, bp: 73.0, s1: 193, s2: 128.0, s3: 50, s4: 4, s5: 4.5, s6: 92, progression: 125 },
    ],
  },
}

// Feature types for data quality analysis
const getFeatureType = (feature: string, data: any[]) => {
  // Check if feature is categorical
  if (feature === "species" || feature === "sex" || feature === "chas" || feature === "rad") {
    return "categorical"
  }

  // Check if feature has few unique values
  const uniqueValues = new Set(data.map((item) => item[feature]))
  if (uniqueValues.size <= 10) {
    return "categorical"
  }

  return "numerical"
}

// Helper functions for statistics
const calculateMean = (data: number[]): number => {
  return data.reduce((sum, val) => sum + val, 0) / data.length
}

const calculateMedian = (data: number[]): number => {
  const sorted = [...data].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

const calculateMode = (data: number[]): number => {
  const counts: Record<number, number> = {}
  let maxCount = 0
  let mode = data[0]

  data.forEach((val) => {
    counts[val] = (counts[val] || 0) + 1
    if (counts[val] > maxCount) {
      maxCount = counts[val]
      mode = val
    }
  })

  return mode
}

const calculateStdDev = (data: number[], mean: number): number => {
  const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length
  return Math.sqrt(variance)
}

const calculateQuantile = (data: number[], q: number): number => {
  const sorted = [...data].sort((a, b) => a - b)
  const pos = (sorted.length - 1) * q
  const base = Math.floor(pos)
  const rest = pos - base

  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base])
  }
  return sorted[base]
}

const calculateCorrelation = (x: number[], y: number[]): number => {
  const n = x.length
  const meanX = calculateMean(x)
  const meanY = calculateMean(y)

  let numerator = 0
  let denomX = 0
  let denomY = 0

  for (let i = 0; i < n; i++) {
    numerator += (x[i] - meanX) * (y[i] - meanY)
    denomX += Math.pow(x[i] - meanX, 2)
    denomY += Math.pow(y[i] - meanY, 2)
  }

  return numerator / (Math.sqrt(denomX) * Math.sqrt(denomY))
}

// Function to detect outliers using IQR method
const detectOutliers = (data: number[]): number[] => {
  const q1 = calculateQuantile(data, 0.25)
  const q3 = calculateQuantile(data, 0.75)
  const iqr = q3 - q1
  const lowerBound = q1 - 1.5 * iqr
  const upperBound = q3 + 1.5 * iqr

  return data.filter((val) => val < lowerBound || val > upperBound)
}

// Function to generate histogram data
const generateHistogram = (data: number[], bins = 10): { labels: string[]; values: number[] } => {
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min
  const binWidth = range / bins

  const histValues = Array(bins).fill(0)
  const histLabels = Array(bins)
    .fill(0)
    .map((_, i) => {
      const binStart = min + i * binWidth
      const binEnd = min + (i + 1) * binWidth
      return `${binStart.toFixed(1)}-${binEnd.toFixed(1)}`
    })

  data.forEach((val) => {
    if (val === max) {
      histValues[bins - 1]++
      return
    }

    const binIndex = Math.floor((val - min) / binWidth)
    histValues[binIndex]++
  })

  return { labels: histLabels, values: histValues }
}

export function EnhancedEdaDemo() {
  // Dataset selection
  const [selectedDataset, setSelectedDataset] = useState<string>("iris")
  const [activeTab, setActiveTab] = useState<string>("overview")
  const [activeFeature, setActiveFeature] = useState<string>("")
  const [secondFeature, setSecondFeature] = useState<string>("")
  const [targetFeature, setTargetFeature] = useState<string>("")

  // Data modification parameters
  const [outlierPercentage, setOutlierPercentage] = useState<number>(0)
  const [missingValuePercentage, setMissingValuePercentage] = useState<number>(0)
  const [noiseLevel, setNoiseLevel] = useState<number>(0)

  // Visualization parameters
  const [binCount, setBinCount] = useState<number>(10)
  const [showMean, setShowMean] = useState<boolean>(true)
  const [showMedian, setShowMedian] = useState<boolean>(true)
  const [showOutliers, setShowOutliers] = useState<boolean>(true)

  // Data state
  const [originalData, setOriginalData] = useState<any[]>([])
  const [workingData, setWorkingData] = useState<any[]>([])
  const [features, setFeatures] = useState<string[]>([])
  const [dataStats, setDataStats] = useState<any>({})
  const [dataQuality, setDataQuality] = useState<any>({})
  const [correlationMatrix, setCorrelationMatrix] = useState<any[]>([])

  // Initialize data
  useEffect(() => {
    const dataset = DATASETS[selectedDataset as keyof typeof DATASETS]
    setOriginalData(dataset.data)
    setWorkingData(dataset.data)
    setFeatures(dataset.features)
    setActiveFeature(dataset.features[0])
    setSecondFeature(dataset.features[1])
    setTargetFeature(dataset.target)

    // Calculate initial statistics
    calculateDataStats(dataset.data, dataset.features, dataset.target)
    analyzeDataQuality(dataset.data, dataset.features)
    calculateCorrelationMatrix(dataset.data, dataset.features)
  }, [selectedDataset])

  // Update data when modification parameters change
  useEffect(() => {
    if (originalData.length === 0) return

    // Start with original data
    let newData = [...originalData]

    // Add missing values
    if (missingValuePercentage > 0) {
      const totalCells = newData.length * features.length
      const cellsToNull = Math.floor((totalCells * missingValuePercentage) / 100)

      for (let i = 0; i < cellsToNull; i++) {
        const randomRow = Math.floor(Math.random() * newData.length)
        const randomFeature = features[Math.floor(Math.random() * features.length)]

        // Create a deep copy of the row to modify
        const rowCopy = { ...newData[randomRow] }
        rowCopy[randomFeature] = null

        // Replace the row in the dataset
        newData[randomRow] = rowCopy
      }
    }

    // Add outliers
    if (outlierPercentage > 0) {
      const rowsToModify = Math.floor((newData.length * outlierPercentage) / 100)

      for (let i = 0; i < rowsToModify; i++) {
        const randomRow = Math.floor(Math.random() * newData.length)
        const randomFeature = features[Math.floor(Math.random() * features.length)]

        // Only modify numerical features
        if (getFeatureType(randomFeature, originalData) === "numerical") {
          // Get the range of values for this feature
          const featureValues = originalData.map((row) => row[randomFeature])
          const min = Math.min(...featureValues)
          const max = Math.max(...featureValues)
          const range = max - min

          // Create a deep copy of the row to modify
          const rowCopy = { ...newData[randomRow] }

          // 50% chance of high outlier, 50% chance of low outlier
          if (Math.random() < 0.5) {
            rowCopy[randomFeature] = max + range * (1 + Math.random())
          } else {
            rowCopy[randomFeature] = min - range * Math.random()
          }

          // Replace the row in the dataset
          newData[randomRow] = rowCopy
        }
      }
    }

    // Add noise
    if (noiseLevel > 0) {
      newData = newData.map((row) => {
        const newRow = { ...row }

        features.forEach((feature) => {
          // Only add noise to numerical features
          if (getFeatureType(feature, originalData) === "numerical" && newRow[feature] !== null) {
            
            const values = originalData.map((row) => row[feature]).filter((value) => value !== null && value !== undefined);
            const originalValues = originalData.map((r) => r[feature])
            const stdDev = calculateStdDev(originalValues, calculateMean(originalValues))

            // Add random noise based on the standard deviation and noise level
            const noise = (Math.random() * 2 - 1) * stdDev * (noiseLevel / 100)
            newRow[feature] += noise
          }
        })

        return newRow
      })
    }

    setWorkingData(newData)

    // Recalculate statistics with modified data
    calculateDataStats(newData, features, targetFeature)
    analyzeDataQuality(newData, features)
    calculateCorrelationMatrix(newData, features)
  }, [originalData, features, outlierPercentage, missingValuePercentage, noiseLevel])

  // Calculate statistics for the dataset
  const calculateDataStats = (data: any[], features: string[], target: string) => {
    const stats: any = {
      rowCount: data.length,
      featureCount: features.length,
      features: {},
    }

    // Calculate statistics for each feature
    features.forEach((feature) => {
      const values = data.map((row) => row[feature]).filter((val) => val !== null && val !== undefined)

      if (values.length === 0) {
        stats.features[feature] = { empty: true }
        return
      }

      const featureType = getFeatureType(feature, data)

      if (featureType === "numerical") {
        const mean = calculateMean(values)
        const median = calculateMedian(values)
        const mode = calculateMode(values)
        const stdDev = calculateStdDev(values, mean)
        const min = Math.min(...values)
        const max = Math.max(...values)
        const q1 = calculateQuantile(values, 0.25)
        const q3 = calculateQuantile(values, 0.75)
        const outliers = detectOutliers(values)

        stats.features[feature] = {
          type: "numerical",
          mean,
          median,
          mode,
          stdDev,
          min,
          max,
          q1,
          q3,
          outliers: outliers.length,
          outlierPercentage: (outliers.length / values.length) * 100,
          histogram: generateHistogram(values, binCount),
        }
      } else {
        // Categorical feature
        const valueCounts: Record<string, number> = {}
        values.forEach((val) => {
          valueCounts[val] = (valueCounts[val] || 0) + 1
        })

        const uniqueValues = Object.keys(valueCounts)

        stats.features[feature] = {
          type: "categorical",
          uniqueValues: uniqueValues.length,
          mostCommon: Object.entries(valueCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([val, count]) => ({ value: val, count })),
          valueCounts,
        }
      }
    })

    // Target variable statistics
    if (target) {
      const targetValues = data.map((row) => row[target]).filter((val) => val !== null && val !== undefined)
      const targetType = getFeatureType(target, data)

      if (targetType === "numerical") {
        stats.target = {
          type: "numerical",
          mean: calculateMean(targetValues),
          median: calculateMedian(targetValues),
          stdDev: calculateStdDev(targetValues, calculateMean(targetValues)),
          min: Math.min(...targetValues),
          max: Math.max(...targetValues),
        }
      } else {
        const valueCounts: Record<string, number> = {}
        targetValues.forEach((val) => {
          valueCounts[val] = (valueCounts[val] || 0) + 1
        })

        stats.target = {
          type: "categorical",
          uniqueValues: Object.keys(valueCounts).length,
          valueCounts,
        }
      }
    }

    setDataStats(stats)
  }

  // Analyze data quality
  const analyzeDataQuality = (data: any[], features: string[]) => {
    const quality: any = {
      missingValues: {},
      duplicateRows: 0,
      outliers: {},
    }

    // Check for missing values
    features.forEach((feature) => {
      const missingCount = data.filter((row) => row[feature] === null || row[feature] === undefined).length
      quality.missingValues[feature] = {
        count: missingCount,
        percentage: (missingCount / data.length) * 100,
      }
    })

    // Check for duplicate rows
    const stringifiedRows = data.map((row) => JSON.stringify(row))
    const uniqueRows = new Set(stringifiedRows)
    quality.duplicateRows = data.length - uniqueRows.size
    quality.duplicatePercentage = ((data.length - uniqueRows.size) / data.length) * 100

    // Check for outliers in numerical features
    features.forEach((feature) => {
      if (getFeatureType(feature, data) === "numerical") {
        const values = data.map((row) => row[feature]).filter((val) => val !== null && val !== undefined)

        const outliers = detectOutliers(values)

        quality.outliers[feature] = {
          count: outliers.length,
          percentage: (outliers.length / values.length) * 100,
          values: outliers.slice(0, 5), // Show only first 5 outliers
        }
      }
    })

    setDataQuality(quality)
  }

  // Calculate correlation matrix
  const calculateCorrelationMatrix = (data: any[], features: string[]) => {
    const numericalFeatures = features.filter((feature) => getFeatureType(feature, data) === "numerical")

    const matrix: any[] = []

    numericalFeatures.forEach((feature1) => {
      const row: any = { feature: feature1 }

      numericalFeatures.forEach((feature2) => {
        const values1 = data.map((row) => row[feature1]).filter((val) => val !== null && val !== undefined)

        const values2 = data.map((row) => row[feature2]).filter((val) => val !== null && val !== undefined)

        // Only calculate correlation if we have the same number of valid values
        if (values1.length === values2.length && values1.length > 0) {
          row[feature2] = calculateCorrelation(values1, values2)
        } else {
          row[feature2] = null
        }
      })

      matrix.push(row)
    })

    setCorrelationMatrix(matrix)
  }

  // Reset to original data
  const handleReset = () => {
    setOutlierPercentage(0)
    setMissingValuePercentage(0)
    setNoiseLevel(0)
    setBinCount(10)
    setWorkingData([...originalData])
    calculateDataStats(originalData, features, targetFeature)
    analyzeDataQuality(originalData, features)
    calculateCorrelationMatrix(originalData, features)
  }

  // Prepare chart data for univariate analysis
  const getUnivariatePlotData = () => {
    if (!activeFeature || !dataStats.features || !dataStats.features[activeFeature]) {
      return null
    }

    const featureStats = dataStats.features[activeFeature]

    if (featureStats.type === "numerical") {
      // Histogram for numerical features
      return {
        labels: featureStats.histogram.labels,
        datasets: [
          {
            label: activeFeature,
            data: featureStats.histogram.values,
            backgroundColor: "rgba(53, 162, 235, 0.5)",
            borderColor: "rgba(53, 162, 235, 1)",
            borderWidth: 1,
          },
        ],
      }
    } else {
      // Bar chart for categorical features
      const counts = featureStats.valueCounts
      return {
        labels: Object.keys(counts),
        datasets: [
          {
            label: activeFeature,
            data: Object.values(counts),
            backgroundColor: "rgba(53, 162, 235, 0.5)",
            borderColor: "rgba(53, 162, 235, 1)",
            borderWidth: 1,
          },
        ],
      }
    }
  }

  // Prepare chart data for bivariate analysis
  const getBivariatePlotData = () => {
    if (!activeFeature || !secondFeature || !workingData.length) {
      return null
    }

    const feature1Type = getFeatureType(activeFeature, workingData)
    const feature2Type = getFeatureType(secondFeature, workingData)

    // Filter out rows with missing values
    const filteredData = workingData.filter(
      (row) =>
        row[activeFeature] !== null &&
        row[activeFeature] !== undefined &&
        row[secondFeature] !== null &&
        row[secondFeature] !== undefined,
    )

    if (feature1Type === "numerical" && feature2Type === "numerical") {
      // Scatter plot for two numerical features
      return {
        datasets: [
          {
            label: `${activeFeature} vs ${secondFeature}`,
            data: filteredData.map((row) => ({
              x: row[activeFeature],
              y: row[secondFeature],
            })),
            backgroundColor: "rgba(53, 162, 235, 0.5)",
            borderColor: "rgba(53, 162, 235, 1)",
            borderWidth: 1,
            pointRadius: 5,
            pointHoverRadius: 7,
          },
        ],
      }
    } else if (feature1Type === "categorical" && feature2Type === "numerical") {
      // Box plot data for categorical vs numerical
      const categories = [...new Set(filteredData.map((row) => row[activeFeature]))]

      return {
        labels: categories,
        datasets: categories.map((category, index) => {
          const categoryData = filteredData
            .filter((row) => row[activeFeature] === category)
            .map((row) => row[secondFeature])

          const mean = calculateMean(categoryData)
          const median = calculateMedian(categoryData)
          const q1 = calculateQuantile(categoryData, 0.25)
          const q3 = calculateQuantile(categoryData, 0.75)
          const min = Math.min(...categoryData)
          const max = Math.max(...categoryData)

          return {
            label: category.toString(),
            data: [{ min, q1, median, q3, max }],
            backgroundColor: `hsla(${index * 50}, 70%, 60%, 0.5)`,
            borderColor: `hsla(${index * 50}, 70%, 60%, 1)`,
            borderWidth: 1,
          }
        }),
      }
    } else if (feature1Type === "numerical" && feature2Type === "categorical") {
      // Swap features for box plot
      const categories = [...new Set(filteredData.map((row) => row[secondFeature]))]

      return {
        labels: categories,
        datasets: categories.map((category, index) => {
          const categoryData = filteredData
            .filter((row) => row[secondFeature] === category)
            .map((row) => row[activeFeature])

          const mean = calculateMean(categoryData)
          const median = calculateMedian(categoryData)
          const q1 = calculateQuantile(categoryData, 0.25)
          const q3 = calculateQuantile(categoryData, 0.75)
          const min = Math.min(...categoryData)
          const max = Math.max(...categoryData)

          return {
            label: category.toString(),
            data: [{ min, q1, median, q3, max }],
            backgroundColor: `hsla(${index * 50}, 70%, 60%, 0.5)`,
            borderColor: `hsla(${index * 50}, 70%, 60%, 1)`,
            borderWidth: 1,
          }
        }),
      }
    } else {
      // Heatmap for two categorical features
      const categories1 = [...new Set(filteredData.map((row) => row[activeFeature]))]
      const categories2 = [...new Set(filteredData.map((row) => row[secondFeature]))]

      const counts: number[][] = Array(categories1.length)
        .fill(0)
        .map(() => Array(categories2.length).fill(0))

      filteredData.forEach((row) => {
        const i = categories1.indexOf(row[activeFeature])
        const j = categories2.indexOf(row[secondFeature])
        counts[i][j]++
      })

      return {
        labels: categories2,
        datasets: categories1.map((category, index) => ({
          label: category.toString(),
          data: counts[index],
          backgroundColor: `hsla(${index * 50}, 70%, 60%, 0.5)`,
          borderColor: `hsla(${index * 50}, 70%, 60%, 1)`,
          borderWidth: 1,
        })),
      }
    }
  }

  // Prepare chart data for target analysis
  const getTargetAnalysisData = () => {
    if (!activeFeature || !targetFeature || !workingData.length) {
      return null
    }

    const featureType = getFeatureType(activeFeature, workingData)
    const targetType = getFeatureType(targetFeature, workingData)

    // Filter out rows with missing values
    const filteredData = workingData.filter(
      (row) =>
        row[activeFeature] !== null &&
        row[activeFeature] !== undefined &&
        row[targetFeature] !== null &&
        row[targetFeature] !== undefined,
    )

    if (featureType === "numerical" && targetType === "numerical") {
      // Scatter plot for numerical feature vs numerical target
      return {
        datasets: [
          {
            label: `${activeFeature} vs ${targetFeature}`,
            data: filteredData.map((row) => ({
              x: row[activeFeature],
              y: row[targetFeature],
            })),
            backgroundColor: "rgba(255, 99, 132, 0.5)",
            borderColor: "rgba(255, 99, 132, 1)",
            borderWidth: 1,
            pointRadius: 5,
            pointHoverRadius: 7,
          },
        ],
      }
        } else if (featureType === "numerical" && targetType === "categorical") {
          // Bar chart for numerical feature by categorical target using median values
          const categories = [...new Set(filteredData.map((row) => row[targetFeature]))]
     
          return {
            labels: categories,
            datasets: categories.map((category, index) => {
              const categoryData = filteredData
                .filter((row) => row[targetFeature] === category)
                .map((row) => row[activeFeature])
     
              const median = calculateMedian(categoryData)
     
              return {
                label: category.toString(),
                data: [{ x: category, y: median }],
                backgroundColor: `hsla(${index * 50}, 70%, 60%, 0.5)`,
                borderColor: `hsla(${index * 50}, 70%, 60%, 1)`,
                borderWidth: 1,
              }
            }),
          }
    } else if (featureType === "categorical" && targetType === "numerical") {
      // Bar chart with error bars for categorical feature vs numerical target
      const categories = [...new Set(filteredData.map((row) => row[activeFeature]))]

      const means = categories.map((category) => {
        const values = filteredData.filter((row) => row[activeFeature] === category).map((row) => row[targetFeature])

        return calculateMean(values)
      })

      const stdDevs = categories.map((category, index) => {
        const values = filteredData.filter((row) => row[activeFeature] === category).map((row) => row[targetFeature])

        return calculateStdDev(values, means[index])
      })

      return {
        labels: categories,
        datasets: [
          {
            label: `Mean ${targetFeature} by ${activeFeature}`,
            data: means,
            backgroundColor: "rgba(255, 99, 132, 0.5)",
            borderColor: "rgba(255, 99, 132, 1)",
            borderWidth: 1,
          },
        ],
      }
    } else {
      // Stacked bar chart for categorical feature vs categorical target
      const featureCategories = [...new Set(filteredData.map((row) => row[activeFeature]))]
      const targetCategories = [...new Set(filteredData.map((row) => row[targetFeature]))]

      return {
        labels: featureCategories,
        datasets: targetCategories.map((targetCategory, index) => {
          const counts = featureCategories.map((featureCategory) => {
            return filteredData.filter(
              (row) => row[activeFeature] === featureCategory && row[targetFeature] === targetCategory,
            ).length
          })

          return {
            label: targetCategory.toString(),
            data: counts,
            backgroundColor: `hsla(${index * 50}, 70%, 60%, 0.5)`,
            borderColor: `hsla(${index * 50}, 70%, 60%, 1)`,
            borderWidth: 1,
          }
        }),
      }
    }
  }

  // Get correlation heatmap data
  const getCorrelationHeatmapData = () => {
    if (!correlationMatrix.length) return null

    const features = correlationMatrix.map((row) => row.feature)

    return {
      labels: features,
      datasets: features.map((feature, index) => {
        const values = features.map((f) => correlationMatrix[index][f] || 0)

        return {
          label: feature,
          data: values,
          backgroundColor: values.map((v) => {
            // Color based on correlation strength
            const absValue = Math.abs(v)
            if (v > 0) {
              return `rgba(0, 0, 255, ${absValue})`
            } else {
              return `rgba(255, 0, 0, ${absValue})`
            }
          }),
          borderColor: "rgba(0, 0, 0, 0.1)",
          borderWidth: 1,
        }
      }),
    }
  }

  return (
    <div className="space-y-6 py-4">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Exploratory Data Analysis</h2>
          <p className="text-muted-foreground">Explore and analyze datasets to discover patterns and insights</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedDataset} onValueChange={setSelectedDataset}>
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select dataset" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="iris">Iris Flower Dataset</SelectItem>
              <SelectItem value="boston">Boston Housing Dataset</SelectItem>
              <SelectItem value="diabetes">Diabetes Dataset</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" onClick={handleReset}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-5">
          <TabsTrigger value="overview">
            <Table2 className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="univariate">
            <BarChart3 className="h-4 w-4 mr-2" />
            Univariate
          </TabsTrigger>
          <TabsTrigger value="bivariate">
            <ScatterChart className="h-4 w-4 mr-2" />
            Bivariate
          </TabsTrigger>
          <TabsTrigger value="multivariate">
            <Sigma className="h-4 w-4 mr-2" />
            Multivariate
          </TabsTrigger>
          <TabsTrigger value="target">
            <Target className="h-4 w-4 mr-2" />
            Target Analysis
          </TabsTrigger>
        </TabsList>

        {/* Data Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="p-4">
              <h3 className="text-lg font-medium mb-2">Dataset Information</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Name:</span>
                  <span className="font-medium">{DATASETS[selectedDataset as keyof typeof DATASETS].name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Rows:</span>
                  <span className="font-medium">{dataStats.rowCount || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Features:</span>
                  <span className="font-medium">{dataStats.featureCount || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Target:</span>
                  <span className="font-medium">{targetFeature}</span>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-4">
                {DATASETS[selectedDataset as keyof typeof DATASETS].description}
              </p>
            </Card>

            <Card className="p-4">
              <h3 className="text-lg font-medium mb-2">Data Quality</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Missing Values:</span>
                  <span className="font-medium">
                    {dataQuality.missingValues
                      ? Object.values(dataQuality.missingValues).reduce((sum: number, val: any) => sum + val.count, 0)
                      : 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Duplicate Rows:</span>
                  <span className="font-medium">{dataQuality.duplicateRows || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Outliers:</span>
                  <span className="font-medium">
                    {dataQuality.outliers
                      ? Object.values(dataQuality.outliers).reduce((sum: number, val: any) => sum + val.count, 0)
                      : 0}
                  </span>
                </div>
              </div>
              <div className="mt-4">
                <h4 className="text-sm font-medium mb-2">Data Modification Controls</h4>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between">
                      <Label htmlFor="missing-values">Missing Values (%)</Label>
                      <span className="text-sm">{missingValuePercentage}%</span>
                    </div>
                    <Slider
                      id="missing-values"
                      value={[missingValuePercentage]}
                      min={0}
                      max={30}
                      step={1}
                      onValueChange={(value) => setMissingValuePercentage(value[0])}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between">
                      <Label htmlFor="outliers">Outliers (%)</Label>
                      <span className="text-sm">{outlierPercentage}%</span>
                    </div>
                    <Slider
                      id="outliers"
                      value={[outlierPercentage]}
                      min={0}
                      max={20}
                      step={1}
                      onValueChange={(value) => setOutlierPercentage(value[0])}
                      className="mt-2"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between">
                      <Label htmlFor="noise">Noise Level (%)</Label>
                      <span className="text-sm">{noiseLevel}%</span>
                    </div>
                    <Slider
                      id="noise"
                      value={[noiseLevel]}
                      min={0}
                      max={50}
                      step={1}
                      onValueChange={(value) => setNoiseLevel(value[0])}
                      className="mt-2"
                    />
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <h3 className="text-lg font-medium mb-2">Feature Types</h3>
              <div className="space-y-2">
                {features.map((feature) => (
                  <div key={feature} className="flex justify-between items-center">
                    <span className="text-muted-foreground">{feature}:</span>
                    <Badge
                      variant={
                        dataStats.features && dataStats.features[feature]?.type === "numerical"
                          ? "default"
                          : "secondary"
                      }
                    >
                      {dataStats.features && dataStats.features[feature]?.type === "numerical"
                        ? "Numerical"
                        : "Categorical"}
                    </Badge>
                  </div>
                ))}
                <Separator className="my-2" />
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">{targetFeature} (target):</span>
                  <Badge variant={dataStats.target && dataStats.target.type === "numerical" ? "default" : "secondary"}>
                    {dataStats.target && dataStats.target.type === "numerical" ? "Numerical" : "Categorical"}
                  </Badge>
                </div>
              </div>
            </Card>
          </div>

          <Card className="p-4">
            <h3 className="text-lg font-medium mb-4">Data Preview</h3>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-muted">
                    <th className="p-2 text-left">#</th>
                    {features.map((feature) => (
                      <th key={feature} className="p-2 text-left">
                        {feature}
                      </th>
                    ))}
                    <th className="p-2 text-left">{targetFeature} (target)</th>
                  </tr>
                </thead>
                <tbody>
                  {workingData.slice(0, 10).map((row, index) => (
                    <tr key={index} className="border-b border-muted">
                      <td className="p-2">{index + 1}</td>
                      {features.map((feature) => (
                        <td key={feature} className="p-2">
                          {row[feature] === null || row[feature] === undefined ? (
                            <span className="text-muted-foreground italic">null</span>
                          ) : typeof row[feature] === "number" ? (
                            row[feature].toFixed(2)
                          ) : (
                            row[feature]
                          )}
                        </td>
                      ))}
                      <td className="p-2 font-medium">
                        {typeof row[targetFeature] === "number" ? row[targetFeature].toFixed(2) : row[targetFeature]}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-muted-foreground mt-2">Showing 10 of {workingData.length} rows</p>
          </Card>

          <Card className="p-4">
            <h3 className="text-lg font-medium mb-4">Missing Values</h3>
            <div className="space-y-4">
              {features.map((feature) => {
                const missingInfo = dataQuality.missingValues && dataQuality.missingValues[feature]
                const missingPercentage = missingInfo ? missingInfo.percentage : 0

                return (
                  <div key={feature} className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-sm">{feature}</span>
                      <span className="text-sm text-muted-foreground">
                        {missingInfo ? missingInfo.count : 0} rows ({missingPercentage.toFixed(1)}%)
                      </span>
                    </div>
                    <Progress value={missingPercentage} className="h-2" />
                  </div>
                )
              })}
            </div>
          </Card>
        </TabsContent>

        {/* Univariate Analysis Tab */}
        <TabsContent value="univariate" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2">
              <Card className="p-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">Distribution of {activeFeature}</h3>
                  <Select value={activeFeature} onValueChange={setActiveFeature}>
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Select feature" />
                    </SelectTrigger>
                    <SelectContent>
                      {features.map((feature) => (
                        <SelectItem key={feature} value={feature}>
                          {feature}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="aspect-[16/9] bg-muted rounded-md p-4">
                  {dataStats.features &&
                    dataStats.features[activeFeature] &&
                    (dataStats.features[activeFeature].type === "numerical" ? (
                      <Bar
                        data={getUnivariatePlotData() || { labels: [], datasets: [] }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: true,
                          plugins: {
                            legend: {
                              display: false,
                            },
                            title: {
                              display: true,
                              text: `Distribution of ${activeFeature}`,
                            },
                            tooltip: {
                              callbacks: {
                                title: (context) => `Range: ${context[0].label}`,
                                label: (context) => `Count: ${context.raw}`,
                              },
                            },
                          },
                          scales: {
                            y: {
                              beginAtZero: true,
                              title: {
                                display: true,
                                text: "Frequency",
                              },
                            },
                            x: {
                              title: {
                                display: true,
                                text: activeFeature,
                              },
                            },
                          },
                        }}
                      />
                    ) : (
                      <Bar
                        data={getUnivariatePlotData() || { labels: [], datasets: [] }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: true,
                          plugins: {
                            legend: {
                              display: false,
                            },
                            title: {
                              display: true,
                              text: `Distribution of ${activeFeature}`,
                            },
                          },
                          scales: {
                            y: {
                              beginAtZero: true,
                              title: {
                                display: true,
                                text: "Count",
                              },
                            },
                            x: {
                              title: {
                                display: true,
                                text: activeFeature,
                              },
                            },
                          },
                        }}
                      />
                    ))}
                </div>

                {dataStats.features &&
                  dataStats.features[activeFeature] &&
                  dataStats.features[activeFeature].type === "numerical" && (
                    <div className="mt-4 flex justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <Checkbox
                            id="show-mean"
                            checked={showMean}
                            onCheckedChange={(checked) => setShowMean(checked as boolean)}
                          />
                          <Label htmlFor="show-mean">Show Mean</Label>
                        </div>
                        <div className="flex items-center gap-2 mt-2">
                          <Checkbox
                            id="show-median"
                            checked={showMedian}
                            onCheckedChange={(checked) => setShowMedian(checked as boolean)}
                          />
                          <Label htmlFor="show-median">Show Median</Label>
                        </div>
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <Checkbox
                            id="show-outliers"
                            checked={showOutliers}
                            onCheckedChange={(checked) => setShowOutliers(checked as boolean)}
                          />
                          <Label htmlFor="show-outliers">Show Outliers</Label>
                        </div>
                        <div className="flex items-center gap-2 mt-2">
                          <Label htmlFor="bin-count">Bins:</Label>
                          <Select
                            value={binCount.toString()}
                            onValueChange={(value) => setBinCount(Number.parseInt(value))}
                          >
                            <SelectTrigger className="w-[80px]">
                              <SelectValue placeholder="Bins" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="5">5</SelectItem>
                              <SelectItem value="10">10</SelectItem>
                              <SelectItem value="15">15</SelectItem>
                              <SelectItem value="20">20</SelectItem>
                              <SelectItem value="30">30</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </div>
                  )}
              </Card>
            </div>

            <Card className="p-4">
              <h3 className="text-lg font-medium mb-4">Feature Statistics</h3>
              {dataStats.features &&
                dataStats.features[activeFeature] &&
                (dataStats.features[activeFeature].type === "numerical" ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Mean</div>
                        <div className="text-lg font-semibold">{dataStats.features[activeFeature].mean.toFixed(2)}</div>
                      </div>
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Median</div>
                        <div className="text-lg font-semibold">
                          {dataStats.features[activeFeature].median.toFixed(2)}
                        </div>
                      </div>
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Std Dev</div>
                        <div className="text-lg font-semibold">
                          {dataStats.features[activeFeature].stdDev.toFixed(2)}
                        </div>
                      </div>
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Range</div>
                        <div className="text-lg font-semibold">
                          {(dataStats.features[activeFeature].max - dataStats.features[activeFeature].min).toFixed(2)}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Five-Number Summary</h4>
                      <div className="relative h-12 bg-muted rounded-md">
                        {/* Box plot visualization */}
                        <div className="absolute top-0 left-0 w-full h-4 flex items-center justify-center">
                          <div className="absolute h-0.5 bg-gray-400 w-full"></div>

                          {/* Min */}
                          <div
                            className="absolute h-4 w-0.5 bg-gray-600"
                            style={{
                              left: `0%`,
                            }}
                          ></div>

                          {/* Q1 */}
                          <div
                            className="absolute h-4 w-0.5 bg-gray-600"
                            style={{
                              left: `${
                                ((dataStats.features[activeFeature].q1 - dataStats.features[activeFeature].min) /
                                  (dataStats.features[activeFeature].max - dataStats.features[activeFeature].min)) *
                                100
                              }%`,
                            }}
                          ></div>

                          {/* Box from Q1 to Q3 */}
                          <div
                            className="absolute h-4 bg-blue-200 border border-blue-400"
                            style={{
                              left: `${
                                ((dataStats.features[activeFeature].q1 - dataStats.features[activeFeature].min) /
                                  (dataStats.features[activeFeature].max - dataStats.features[activeFeature].min)) *
                                100
                              }%`,
                              width: `${
                                ((dataStats.features[activeFeature].q3 - dataStats.features[activeFeature].q1) /
                                  (dataStats.features[activeFeature].max - dataStats.features[activeFeature].min)) *
                                100
                              }%`,
                            }}
                          ></div>

                          {/* Median */}
                          <div
                            className="absolute h-4 w-0.5 bg-blue-600"
                            style={{
                              left: `${
                                ((dataStats.features[activeFeature].median - dataStats.features[activeFeature].min) /
                                  (dataStats.features[activeFeature].max - dataStats.features[activeFeature].min)) *
                                100
                              }%`,
                            }}
                          ></div>

                          {/* Q3 */}
                          <div
                            className="absolute h-4 w-0.5 bg-gray-600"
                            style={{
                              left: `${
                                ((dataStats.features[activeFeature].q3 - dataStats.features[activeFeature].min) /
                                  (dataStats.features[activeFeature].max - dataStats.features[activeFeature].min)) *
                                100
                              }%`,
                            }}
                          ></div>

                          {/* Max */}
                          <div
                            className="absolute h-4 w-0.5 bg-gray-600"
                            style={{
                              left: `100%`,
                            }}
                          ></div>
                        </div>

                        {/* Labels */}
                        <div className="absolute top-6 left-0 w-full flex justify-between text-xs text-gray-500">
                          <div className="transform -translate-x-1/2">
                            {dataStats.features[activeFeature].min.toFixed(1)}
                          </div>
                          <div
                            className="transform -translate-x-1/2"
                            style={{
                              left: `${
                                ((dataStats.features[activeFeature].median - dataStats.features[activeFeature].min) /
                                  (dataStats.features[activeFeature].max - dataStats.features[activeFeature].min)) *
                                100
                              }%`,
                            }}
                          >
                            {dataStats.features[activeFeature].median.toFixed(1)}
                          </div>
                          <div className="transform -translate-x-1/2">
                            {dataStats.features[activeFeature].max.toFixed(1)}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <h4 className="text-sm font-medium">Outliers</h4>
                        <Badge variant="outline">
                          {dataStats.features[activeFeature].outliers} (
                          {dataStats.features[activeFeature].outlierPercentage.toFixed(1)}%)
                        </Badge>
                      </div>
                      {dataStats.features[activeFeature].outliers > 0 &&
                        dataQuality.outliers &&
                        dataQuality.outliers[activeFeature] && (
                          <div className="text-sm text-muted-foreground">
                            Sample outliers:{" "}
                            {dataQuality.outliers[activeFeature].values.map((v: number) => v.toFixed(2)).join(", ")}
                          </div>
                        )}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Unique Values</div>
                        <div className="text-lg font-semibold">{dataStats.features[activeFeature].uniqueValues}</div>
                      </div>
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Most Common</div>
                        <div className="text-lg font-semibold">
                          {dataStats.features[activeFeature].mostCommon[0].value}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Value Counts</h4>
                      <div className="space-y-2">
                        {Object.entries(dataStats.features[activeFeature].valueCounts).map(
                          ([value, count]: [string, any]) => (
                            <div key={value} className="flex justify-between items-center">
                              <span className="text-sm">{value}</span>
                              <span className="text-sm text-muted-foreground">
                                {count} ({((count / workingData.length) * 100).toFixed(1)}%)
                              </span>
                            </div>
                          ),
                        )}
                      </div>
                    </div>
                  </div>
                ))}
            </Card>
          </div>
        </TabsContent>

        {/* Bivariate Analysis Tab */}
        <TabsContent value="bivariate" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2">
              <Card className="p-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">
                    Relationship: {activeFeature} vs {secondFeature}
                  </h3>
                  <div className="flex gap-2">
                    <Select value={activeFeature} onValueChange={setActiveFeature}>
                      <SelectTrigger className="w-[150px]">
                        <SelectValue placeholder="First feature" />
                      </SelectTrigger>
                      <SelectContent>
                        {features.map((feature) => (
                          <SelectItem key={feature} value={feature}>
                            {feature}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select value={secondFeature} onValueChange={setSecondFeature}>
                      <SelectTrigger className="w-[150px]">
                        <SelectValue placeholder="Second feature" />
                      </SelectTrigger>
                      <SelectContent>
                        {features.map((feature) => (
                          <SelectItem key={feature} value={feature}>
                            {feature}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="aspect-[16/9] bg-muted rounded-md p-4">
                  {activeFeature &&
                  secondFeature &&
                  (getFeatureType(activeFeature, workingData) === "numerical" &&
                  getFeatureType(secondFeature, workingData) === "numerical" ? (
                    <Scatter
                      data={getBivariatePlotData() as ChartData<"scatter", { x: any; y: any; }[], unknown> || { datasets: [] }}
                      options={{
                          responsive: true,
                          maintainAspectRatio: true,
                          plugins: {
                            legend: {
                              display: false,
                            },
                            title: {
                              display: true,
                              text: `${activeFeature} vs ${secondFeature}`,
                            },
                          },
                          scales: {
                            y: {
                              title: {
                                display: true,
                                text: secondFeature,
                              },
                            },
                            x: {
                              title: {
                                display: true,
                                text: activeFeature,
                              },
                            },
                          },
                        }}
                      />
                    ) : (
                      <Bar
                        data={getBivariatePlotData() as ChartData<"bar", any[], unknown> || { labels: [], datasets: [] }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: true,
                          plugins: {
                            title: {
                              display: true,
                              text: `${activeFeature} vs ${secondFeature}`,
                            },
                          },
                          scales: {
                            y: {
                              beginAtZero: true,
                              title: {
                                display: true,
                                text: "Value",
                              },
                            },
                            x: {
                              title: {
                                display: true,
                                text: "Category",
                              },
                            },
                          },
                        }}
                      />
                    ))}
                </div>
              </Card>
            </div>

            <Card className="p-4">
              <h3 className="text-lg font-medium mb-4">Relationship Analysis</h3>
              {activeFeature &&
                secondFeature &&
                (getFeatureType(activeFeature, workingData) === "numerical" &&
                getFeatureType(secondFeature, workingData) === "numerical" ? (
                  <div className="space-y-4">
                    <div className="bg-muted p-3 rounded-md">
                      <div className="text-sm text-muted-foreground">Correlation</div>
                      <div className="text-2xl font-semibold">
                        {(() => {
                          // Calculate correlation between the two features
                          const filteredData = workingData.filter(
                            (row) =>
                              row[activeFeature] !== null &&
                              row[activeFeature] !== undefined &&
                              row[secondFeature] !== null &&
                              row[secondFeature] !== undefined,
                          )

                          const values1 = filteredData.map((row) => row[activeFeature])
                          const values2 = filteredData.map((row) => row[secondFeature])

                          return calculateCorrelation(values1, values2).toFixed(2)
                        })()}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Interpretation</h4>
                      <div className="text-sm text-muted-foreground">
                        {(() => {
                          // Calculate correlation between the two features
                          const filteredData = workingData.filter(
                            (row) =>
                              row[activeFeature] !== null &&
                              row[activeFeature] !== undefined &&
                              row[secondFeature] !== null &&
                              row[secondFeature] !== undefined,
                          )

                          const values1 = filteredData.map((row) => row[activeFeature])
                          const values2 = filteredData.map((row) => row[secondFeature])

                          const corr = calculateCorrelation(values1, values2)
                          const absCorr = Math.abs(corr)

                          if (absCorr < 0.3) {
                            return `Weak ${corr < 0 ? "negative" : "positive"} correlation between ${activeFeature} and ${secondFeature}.`
                          } else if (absCorr < 0.7) {
                            return `Moderate ${corr < 0 ? "negative" : "positive"} correlation between ${activeFeature} and ${secondFeature}.`
                          } else {
                            return `Strong ${corr < 0 ? "negative" : "positive"} correlation between ${activeFeature} and ${secondFeature}.`
                          }
                        })()}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Feature Comparison</h4>
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2">Statistic</th>
                            <th className="text-right py-2">{activeFeature}</th>
                            <th className="text-right py-2">{secondFeature}</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr className="border-b">
                            <td className="py-1">Mean</td>
                            <td className="text-right">{dataStats.features[activeFeature].mean.toFixed(2)}</td>
                            <td className="text-right">{dataStats.features[secondFeature].mean.toFixed(2)}</td>
                          </tr>
                          <tr className="border-b">
                            <td className="py-1">Std Dev</td>
                            <td className="text-right">{dataStats.features[activeFeature].stdDev.toFixed(2)}</td>
                            <td className="text-right">{dataStats.features[secondFeature].stdDev.toFixed(2)}</td>
                          </tr>
                          <tr className="border-b">
                            <td className="py-1">Min</td>
                            <td className="text-right">{dataStats.features[activeFeature].min.toFixed(2)}</td>
                            <td className="text-right">{dataStats.features[secondFeature].min.toFixed(2)}</td>
                          </tr>
                          <tr>
                            <td className="py-1">Max</td>
                            <td className="text-right">{dataStats.features[activeFeature].max.toFixed(2)}</td>
                            <td className="text-right">{dataStats.features[secondFeature].max.toFixed(2)}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="bg-muted p-3 rounded-md">
                      <div className="text-sm text-muted-foreground">Relationship Type</div>
                      <div className="text-lg font-semibold">
                        {getFeatureType(activeFeature, workingData) === "categorical" &&
                        getFeatureType(secondFeature, workingData) === "categorical"
                          ? "Categorical vs Categorical"
                          : getFeatureType(activeFeature, workingData) === "categorical"
                            ? "Categorical vs Numerical"
                            : "Numerical vs Categorical"}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Analysis</h4>
                      <div className="text-sm text-muted-foreground">
                        {getFeatureType(activeFeature, workingData) === "categorical" &&
                        getFeatureType(secondFeature, workingData) === "categorical"
                          ? `This chart shows the relationship between two categorical variables. Look for patterns in how categories from ${activeFeature} relate to categories in ${secondFeature}.`
                          : getFeatureType(activeFeature, workingData) === "categorical"
                            ? `This chart shows how the numerical values of ${secondFeature} vary across different categories of ${activeFeature}. Look for differences in distributions between categories.`
                            : `This chart shows how the numerical values of ${activeFeature} vary across different categories of ${secondFeature}. Look for differences in distributions between categories.`}
                      </div>
                    </div>
                  </div>
                ))}
            </Card>
          </div>
        </TabsContent>

        {/* Multivariate Analysis Tab */}
        <TabsContent value="multivariate" className="space-y-4">
          <Card className="p-4">
            <h3 className="text-lg font-medium mb-4">Correlation Matrix</h3>
            <div className="aspect-[16/9] bg-muted rounded-md p-4">
              <Bar
                data={getCorrelationHeatmapData() || { labels: [], datasets: [] }}
                options={{
                  responsive: true,
                  maintainAspectRatio: true,
                  indexAxis: "y" as const,
                  plugins: {
                    legend: {
                      display: false,
                    },
                    title: {
                      display: true,
                      text: "Feature Correlations",
                    },
                    tooltip: {
                      callbacks: {
                        label: (context) => {
                          const value = context.raw as number
                          return `Correlation: ${value.toFixed(2)}`
                        },
                      },
                    },
                  },
                  scales: {
                    x: {
                      min: -1,
                      max: 1,
                      ticks: {
                        callback: (value) => (typeof value === "number" ? value.toFixed(1) : value),
                      },
                    },
                  },
                }}
              />
            </div>
            <div className="mt-4 text-sm text-muted-foreground">
              <p>
                This correlation matrix shows the Pearson correlation coefficient between pairs of numerical features.
                Values close to 1 indicate strong positive correlation, values close to -1 indicate strong negative
                correlation, and values close to 0 indicate little to no linear correlation.
              </p>
            </div>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="p-4">
              <h3 className="text-lg font-medium mb-4">Feature Importance</h3>
              <div className="space-y-4">
                {features
                  .filter((feature) => getFeatureType(feature, workingData) === "numerical")
                  .map((feature) => {
                    // Calculate a simple "importance" score based on correlation with target
                    // if target is numerical
                    let importance = 0

                    if (dataStats.target && dataStats.target.type === "numerical") {
                      const filteredData = workingData.filter(
                        (row) =>
                          row[feature] !== null &&
                          row[feature] !== undefined &&
                          row[targetFeature] !== null &&
                          row[targetFeature] !== undefined,
                      )

                      const featureValues = filteredData.map((row) => row[feature])
                      const targetValues = filteredData.map((row) => row[targetFeature])

                      importance = Math.abs(calculateCorrelation(featureValues, targetValues))
                    } else {
                      // For categorical targets, use a simple variance-based importance
                      importance =
                        dataStats.features[feature].stdDev /
                        (dataStats.features[feature].max - dataStats.features[feature].min)
                    }

                    return (
                      <div key={feature} className="space-y-1">
                        <div className="flex justify-between">
                          <span className="text-sm">{feature}</span>
                          <span className="text-sm text-muted-foreground">{importance.toFixed(2)}</span>
                        </div>
                        <Progress value={importance * 100} className="h-2" />
                      </div>
                    )
                  })}
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                <p>
                  This is a simplified feature importance visualization based on correlation with the target variable
                  (for regression) or feature variance (for classification). More sophisticated feature importance would
                  require training a model.
                </p>
              </div>
            </Card>

            <Card className="p-4">
              <h3 className="text-lg font-medium mb-4">Feature Engineering Ideas</h3>
              <div className="space-y-2">
                <div className="bg-muted p-3 rounded-md">
                  <h4 className="text-sm font-medium">Potential Transformations</h4>
                  <ul className="list-disc pl-5 text-sm text-muted-foreground mt-2 space-y-1">
                    {features
                      .filter((feature) => getFeatureType(feature, workingData) === "numerical")
                      .flatMap((feature) => {
                        const stats = dataStats.features[feature]
                        const suggestions = []

                        // Suggest log transform for right-skewed data
                        if (stats.mean > stats.median && stats.min > 0) {
                          suggestions.push(`Log transform for ${feature} (right-skewed)`)
                        }

                        // Suggest normalization for features with large ranges
                        if (stats.max - stats.min > 100) {
                          suggestions.push(`Normalize ${feature} (large range)`)
                        }

                        // Suggest binning for features with many unique values
                        if (stats.histogram.values.length > 10) {
                          suggestions.push(`Bin ${feature} into categories`)
                        }

                        return suggestions.map((suggestion, i) => <li key={`${feature}-${i}`}>{suggestion}</li>)
                      })}
                  </ul>
                </div>

                <div className="bg-muted p-3 rounded-md">
                  <h4 className="text-sm font-medium">Potential Interactions</h4>
                  <ul className="list-disc pl-5 text-sm text-muted-foreground mt-2 space-y-1">
                    {correlationMatrix.length > 0 &&
                      correlationMatrix.flatMap((row) => {
                        const feature1 = row.feature

                        // Find highly correlated features
                        const correlatedFeatures = Object.entries(row)
                          .filter(
                            ([key, value]) =>
                              key !== "feature" &&
                              typeof value === "number" &&
                              Math.abs(value) > 0.5 &&
                              Math.abs(value) < 0.95 && // Avoid perfectly correlated features
                              key !== feature1,
                          )
                          .map(([feature2]) => feature2)

                        return correlatedFeatures.map((feature2) => (
                          <li key={`${feature1}-${feature2}`}>
                            Create interaction between {feature1} and {feature2}
                          </li>
                        ))
                      })}
                  </ul>
                </div>
              </div>
            </Card>
          </div>
        </TabsContent>

        {/* Target Analysis Tab */}
        <TabsContent value="target" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2">
              <Card className="p-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">
                    Feature vs Target: {activeFeature} vs {targetFeature}
                  </h3>
                  <Select value={activeFeature} onValueChange={setActiveFeature}>
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Select feature" />
                    </SelectTrigger>
                    <SelectContent>
                      {features.map((feature) => (
                        <SelectItem key={feature} value={feature}>
                          {feature}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="aspect-[16/9] bg-muted rounded-md p-4">
                  {activeFeature &&
                    targetFeature &&
                    (getFeatureType(activeFeature, workingData) === "numerical" &&
                    getFeatureType(targetFeature, workingData) === "numerical" ? (
                      <Scatter
                        data={getTargetAnalysisData() as ChartData<"scatter", { x: any; y: any; }[], unknown> || { datasets: [] }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: true,
                          plugins: {
                            legend: {
                              display: false,
                            },
                            title: {
                              display: true,
                              text: `${activeFeature} vs ${targetFeature}`,
                            },
                          },
                          scales: {
                            y: {
                              title: {
                                display: true,
                                text: targetFeature,
                              },
                            },
                            x: {
                              title: {
                                display: true,
                                text: activeFeature,
                              },
                            },
                          },
                        }}
                      />
                    ) : (
                      <Bar
                        data={getTargetAnalysisData() as ChartData<"bar", number[], unknown> || { labels: [], datasets: [] }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: true,
                          plugins: {
                            title: {
                              display: true,
                              text: `${activeFeature} vs ${targetFeature}`,
                            },
                          },
                          scales: {
                            y: {
                              beginAtZero: true,
                              title: {
                                display: true,
                                text: "Value",
                              },
                            },
                            x: {
                              title: {
                                display: true,
                                text: "Category",
                              },
                            },
                          },
                        }}
                      />
                    ))}
                </div>
              </Card>
            </div>

            <Card className="p-4">
              <h3 className="text-lg font-medium mb-4">Target Variable Analysis</h3>
              {dataStats.target &&
                (dataStats.target.type === "numerical" ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Mean</div>
                        <div className="text-lg font-semibold">{dataStats.target.mean.toFixed(2)}</div>
                      </div>
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Median</div>
                        <div className="text-lg font-semibold">{dataStats.target.median.toFixed(2)}</div>
                      </div>
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Std Dev</div>
                        <div className="text-lg font-semibold">{dataStats.target.stdDev.toFixed(2)}</div>
                      </div>
                      <div className="bg-muted p-2 rounded-md">
                        <div className="text-sm text-muted-foreground">Range</div>
                        <div className="text-lg font-semibold">
                          {(dataStats.target.max - dataStats.target.min).toFixed(2)}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Relationship with {activeFeature}</h4>
                      <div className="text-sm text-muted-foreground">
                        {(() => {
                          // Calculate correlation between feature and target
                          const filteredData = workingData.filter(
                            (row) =>
                              row[activeFeature] !== null &&
                              row[activeFeature] !== undefined &&
                              row[targetFeature] !== null &&
                              row[targetFeature] !== undefined,
                          )

                          const featureValues = filteredData.map((row) => row[activeFeature])
                          const targetValues = filteredData.map((row) => row[targetFeature])

                          const corr = calculateCorrelation(featureValues, targetValues)
                          const absCorr = Math.abs(corr)

                          if (absCorr < 0.3) {
                            return `Weak ${corr < 0 ? "negative" : "positive"} correlation between ${activeFeature} and ${targetFeature}.`
                          } else if (absCorr < 0.7) {
                            return `Moderate ${corr < 0 ? "negative" : "positive"} correlation between ${activeFeature} and ${targetFeature}.`
                          } else {
                            return `Strong ${corr < 0 ? "negative" : "positive"} correlation between ${activeFeature} and ${targetFeature}.`
                          }
                        })()}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="bg-muted p-3 rounded-md">
                      <div className="text-sm text-muted-foreground">Class Distribution</div>
                      <div className="space-y-2 mt-2">
                        {Object.entries(dataStats.target.valueCounts).map(([value, count]: [string, any]) => (
                          <div key={value} className="space-y-1">
                            <div className="flex justify-between">
                              <span className="text-sm">{value}</span>
                              <span className="text-sm text-muted-foreground">
                                {count} ({((count / workingData.length) * 100).toFixed(1)}%)
                              </span>
                            </div>
                            <Progress value={(count / workingData.length) * 100} className="h-2" />
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="text-sm font-medium">Feature Importance for {activeFeature}</h4>
                      <div className="text-sm text-muted-foreground">
                        {getFeatureType(activeFeature, workingData) === "numerical"
                          ? `This numerical feature shows how ${activeFeature} values are distributed across different ${targetFeature} classes.`
                          : `This categorical feature shows the relationship between ${activeFeature} categories and ${targetFeature} classes.`}
                      </div>
                    </div>
                  </div>
                ))}
            </Card>
          </div>

          <Card className="p-4">
            <h3 className="text-lg font-medium mb-4">Predictive Modeling Insights</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Recommended Models</h4>
                <div className="bg-muted p-3 rounded-md">
                  <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                    {dataStats.target && dataStats.target.type === "numerical" ? (
                      <>
                        <li>Linear Regression - For simple linear relationships</li>
                        <li>Random Forest Regressor - For complex non-linear relationships</li>
                        <li>Gradient Boosting Regressor - For high performance</li>
                        <li>Support Vector Regression - For smaller datasets</li>
                      </>
                    ) : (
                      <>
                        <li>Logistic Regression - For simple classification</li>
                        <li>Random Forest Classifier - For complex decision boundaries</li>
                        <li>Gradient Boosting Classifier - For high performance</li>
                        <li>Support Vector Classifier - For smaller datasets</li>
                      </>
                    )}
                  </ul>
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="text-sm font-medium">Preprocessing Recommendations</h4>
                <div className="bg-muted p-3 rounded-md">
                  <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                    <li>Handle missing values ({missingValuePercentage}% in dataset)</li>
                    <li>Remove or cap outliers ({outlierPercentage}% in dataset)</li>
                    <li>Normalize numerical features for distance-based models</li>
                    <li>Encode categorical features appropriately</li>
                    {dataStats.target && dataStats.target.type === "categorical" && (
                      <li>
                        {Object.keys(dataStats.target.valueCounts).length > 2
                          ? "Consider multi-class classification strategies"
                          : "Binary classification problem"}
                      </li>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

