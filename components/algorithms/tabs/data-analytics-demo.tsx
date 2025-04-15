"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { BarChart, Bell, Calculator, Sigma, Dice5, ArrowUpDown } from "lucide-react"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
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
  ArcElement,
} from "chart.js"
import { Bar, Pie, Line } from "react-chartjs-2"

// Register Chart.js components
Chart.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Colors,
  Filler,
)

// Titanic dataset (sample)
const TITANIC_DATASET = {
  name: "Titanic Passenger Dataset",
  description: "Famous dataset containing information about Titanic passengers and their survival",
  features: ["age", "fare", "pclass", "sibsp", "parch", "sex", "embarked"],
  target: "survived",
  size: 100,
  data: [
    { survived: 0, pclass: 3, sex: "male", age: 22.0, sibsp: 1, parch: 0, fare: 7.25, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 38.0, sibsp: 1, parch: 0, fare: 71.28, embarked: "C" },
    { survived: 1, pclass: 3, sex: "female", age: 26.0, sibsp: 0, parch: 0, fare: 7.92, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 35.0, sibsp: 1, parch: 0, fare: 53.1, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 35.0, sibsp: 0, parch: 0, fare: 8.05, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 28.0, sibsp: 0, parch: 0, fare: 8.46, embarked: "Q" },
    { survived: 0, pclass: 1, sex: "male", age: 54.0, sibsp: 0, parch: 0, fare: 51.86, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 2.0, sibsp: 3, parch: 1, fare: 21.08, embarked: "S" },
    { survived: 1, pclass: 3, sex: "female", age: 27.0, sibsp: 0, parch: 2, fare: 11.13, embarked: "S" },
    { survived: 1, pclass: 2, sex: "female", age: 14.0, sibsp: 1, parch: 0, fare: 30.07, embarked: "C" },
    { survived: 1, pclass: 3, sex: "female", age: 4.0, sibsp: 1, parch: 1, fare: 16.7, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 58.0, sibsp: 0, parch: 0, fare: 26.55, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 20.0, sibsp: 0, parch: 0, fare: 8.05, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 39.0, sibsp: 1, parch: 5, fare: 31.28, embarked: "S" },
    { survived: 0, pclass: 3, sex: "female", age: 14.0, sibsp: 0, parch: 0, fare: 7.85, embarked: "S" },
    { survived: 1, pclass: 2, sex: "female", age: 55.0, sibsp: 0, parch: 0, fare: 16.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 2.0, sibsp: 4, parch: 1, fare: 29.13, embarked: "Q" },
    { survived: 1, pclass: 2, sex: "male", age: 23.0, sibsp: 0, parch: 0, fare: 13.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "female", age: 31.0, sibsp: 1, parch: 0, fare: 18.0, embarked: "S" },
    { survived: 1, pclass: 3, sex: "female", age: 45.0, sibsp: 0, parch: 0, fare: 7.23, embarked: "C" },
    { survived: 0, pclass: 3, sex: "male", age: 28.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "Q" },
    { survived: 1, pclass: 2, sex: "female", age: 24.0, sibsp: 0, parch: 0, fare: 13.0, embarked: "S" },
    { survived: 1, pclass: 1, sex: "male", age: 19.0, sibsp: 3, parch: 2, fare: 263.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "female", age: 29.0, sibsp: 0, parch: 4, fare: 21.08, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 65.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "Q" },
    { survived: 0, pclass: 1, sex: "male", age: 28.0, sibsp: 0, parch: 0, fare: 82.17, embarked: "S" },
    { survived: 0, pclass: 2, sex: "male", age: 19.0, sibsp: 0, parch: 0, fare: 10.5, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 17.0, sibsp: 4, parch: 2, fare: 7.92, embarked: "Q" },
    { survived: 1, pclass: 1, sex: "female", age: 26.0, sibsp: 0, parch: 0, fare: 30.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 32.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "Q" },
    { survived: 0, pclass: 1, sex: "male", age: 16.0, sibsp: 0, parch: 0, fare: 26.0, embarked: "S" },
    { survived: 0, pclass: 2, sex: "male", age: 21.0, sibsp: 0, parch: 0, fare: 73.5, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 26.0, sibsp: 1, parch: 0, fare: 7.83, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 32.0, sibsp: 0, parch: 0, fare: 7.93, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 25.0, sibsp: 0, parch: 1, fare: 91.08, embarked: "C" },
    { survived: 0, pclass: 2, sex: "male", age: 0.83, sibsp: 0, parch: 2, fare: 29.0, embarked: "S" },
    { survived: 1, pclass: 3, sex: "female", age: 30.0, sibsp: 0, parch: 0, fare: 7.73, embarked: "Q" },
    { survived: 0, pclass: 1, sex: "male", age: 40.0, sibsp: 0, parch: 0, fare: 27.72, embarked: "C" },
    { survived: 1, pclass: 1, sex: "female", age: 35.0, sibsp: 1, parch: 0, fare: 52.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 24.0, sibsp: 0, parch: 0, fare: 7.05, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 19.0, sibsp: 0, parch: 0, fare: 7.25, embarked: "S" },
    { survived: 1, pclass: 3, sex: "female", age: 29.0, sibsp: 0, parch: 0, fare: 8.14, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 32.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "Q" },
    { survived: 0, pclass: 2, sex: "male", age: 62.0, sibsp: 0, parch: 0, fare: 9.69, embarked: "Q" },
    { survived: 1, pclass: 1, sex: "female", age: 53.0, sibsp: 2, parch: 0, fare: 51.48, embarked: "S" },
    { survived: 1, pclass: 2, sex: "female", age: 36.0, sibsp: 0, parch: 0, fare: 13.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 16.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "Q" },
    { survived: 0, pclass: 3, sex: "male", age: 19.0, sibsp: 0, parch: 0, fare: 8.05, embarked: "S" },
    { survived: 0, pclass: 3, sex: "female", age: 34.0, sibsp: 0, parch: 0, fare: 13.0, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 39.0, sibsp: 1, parch: 0, fare: 55.9, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 32.0, sibsp: 0, parch: 0, fare: 7.85, embarked: "S" },
    { survived: 1, pclass: 3, sex: "female", age: 25.0, sibsp: 1, parch: 0, fare: 7.78, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 17.0, sibsp: 0, parch: 0, fare: 7.05, embarked: "S" },
    { survived: 1, pclass: 2, sex: "female", age: 50.0, sibsp: 0, parch: 1, fare: 26.0, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 48.0, sibsp: 1, parch: 0, fare: 52.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 24.0, sibsp: 0, parch: 0, fare: 7.13, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 29.0, sibsp: 0, parch: 0, fare: 7.88, embarked: "S" },
    { survived: 0, pclass: 1, sex: "male", age: 56.0, sibsp: 0, parch: 0, fare: 30.7, embarked: "C" },
    { survived: 0, pclass: 3, sex: "male", age: 25.0, sibsp: 0, parch: 0, fare: 7.05, embarked: "S" },
    { survived: 0, pclass: 3, sex: "female", age: 33.0, sibsp: 0, parch: 0, fare: 7.73, embarked: "Q" },
    { survived: 0, pclass: 3, sex: "male", age: 22.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "Q" },
    { survived: 0, pclass: 2, sex: "male", age: 28.0, sibsp: 0, parch: 0, fare: 10.5, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 25.0, sibsp: 0, parch: 0, fare: 7.05, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 39.0, sibsp: 0, parch: 0, fare: 24.15, embarked: "C" },
    { survived: 0, pclass: 2, sex: "male", age: 27.0, sibsp: 1, parch: 0, fare: 13.86, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 19.0, sibsp: 0, parch: 0, fare: 30.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 26.0, sibsp: 0, parch: 0, fare: 7.88, embarked: "S" },
    { survived: 1, pclass: 2, sex: "female", age: 32.0, sibsp: 1, parch: 0, fare: 26.0, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 16.0, sibsp: 0, parch: 1, fare: 39.4, embarked: "S" },
    { survived: 1, pclass: 2, sex: "female", age: 30.0, sibsp: 0, parch: 0, fare: 13.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 34.5, sibsp: 0, parch: 0, fare: 6.43, embarked: "C" },
    { survived: 0, pclass: 3, sex: "male", age: 44.0, sibsp: 0, parch: 1, fare: 16.1, embarked: "S" },
    { survived: 1, pclass: 2, sex: "female", age: 18.0, sibsp: 0, parch: 2, fare: 23.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 45.0, sibsp: 0, parch: 0, fare: 7.23, embarked: "C" },
    { survived: 1, pclass: 1, sex: "female", age: 51.0, sibsp: 1, parch: 0, fare: 77.96, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 24.0, sibsp: 0, parch: 0, fare: 7.05, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 41.0, sibsp: 2, parch: 0, fare: 14.1, embarked: "S" },
    { survived: 0, pclass: 2, sex: "male", age: 21.0, sibsp: 1, parch: 0, fare: 11.5, embarked: "S" },
    { survived: 0, pclass: 1, sex: "male", age: 48.0, sibsp: 0, parch: 0, fare: 26.55, embarked: "S" },
    { survived: 0, pclass: 3, sex: "female", age: 22.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "Q" },
    { survived: 1, pclass: 2, sex: "female", age: 24.0, sibsp: 1, parch: 0, fare: 27.0, embarked: "C" },
    { survived: 0, pclass: 3, sex: "male", age: 42.0, sibsp: 0, parch: 0, fare: 7.13, embarked: "S" },
    { survived: 0, pclass: 2, sex: "male", age: 27.0, sibsp: 1, parch: 0, fare: 13.86, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 31.0, sibsp: 0, parch: 0, fare: 28.5, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 30.0, sibsp: 0, parch: 0, fare: 7.23, embarked: "C" },
    { survived: 0, pclass: 3, sex: "male", age: 33.0, sibsp: 0, parch: 0, fare: 7.88, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 44.0, sibsp: 0, parch: 0, fare: 8.05, embarked: "S" },
    { survived: 0, pclass: 2, sex: "male", age: 25.0, sibsp: 0, parch: 0, fare: 13.0, embarked: "S" },
    { survived: 0, pclass: 2, sex: "male", age: 25.0, sibsp: 0, parch: 0, fare: 13.0, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 28.0, sibsp: 0, parch: 0, fare: 9.5, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 21.0, sibsp: 0, parch: 0, fare: 7.78, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 39.0, sibsp: 0, parch: 0, fare: 7.9, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 20.0, sibsp: 0, parch: 0, fare: 7.25, embarked: "S" },
    { survived: 0, pclass: 2, sex: "male", age: 55.0, sibsp: 1, parch: 0, fare: 30.5, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 51.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "S" },
    { survived: 1, pclass: 1, sex: "female", age: 48.0, sibsp: 1, parch: 1, fare: 106.43, embarked: "C" },
    { survived: 0, pclass: 3, sex: "male", age: 29.0, sibsp: 0, parch: 0, fare: 14.46, embarked: "C" },
    { survived: 0, pclass: 1, sex: "male", age: 41.0, sibsp: 0, parch: 0, fare: 26.55, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 21.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "Q" },
    { survived: 0, pclass: 3, sex: "male", age: 29.0, sibsp: 0, parch: 0, fare: 7.23, embarked: "S" },
    { survived: 0, pclass: 3, sex: "female", age: 18.0, sibsp: 0, parch: 0, fare: 7.13, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 24.0, sibsp: 0, parch: 0, fare: 7.25, embarked: "S" },
    { survived: 1, pclass: 3, sex: "female", age: 25.0, sibsp: 0, parch: 0, fare: 7.9, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 28.0, sibsp: 0, parch: 0, fare: 7.9, embarked: "S" },
    { survived: 0, pclass: 3, sex: "male", age: 18.0, sibsp: 0, parch: 0, fare: 7.75, embarked: "S" },
  ],
}

// Helper functions for statistics
const calculateMean = (data: number[]): number => {
  if (data.length === 0) return 0
  return data.reduce((sum, val) => sum + val, 0) / data.length
}

const calculateMedian = (data: number[]): number => {
  if (data.length === 0) return 0
  const sorted = [...data].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

const calculateMode = (data: number[]): number => {
  if (data.length === 0) return 0
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

const calculateVariance = (data: number[], mean: number): number => {
  if (data.length <= 1) return 0
  return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length
}

const calculateStdDev = (data: number[], mean: number): number => {
  return Math.sqrt(calculateVariance(data, mean))
}

const calculateQuantile = (data: number[], q: number): number => {
  if (data.length === 0) return 0
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
  if (x.length !== y.length || x.length === 0) return 0

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

  if (denomX === 0 || denomY === 0) return 0
  return numerator / (Math.sqrt(denomX) * Math.sqrt(denomY))
}

// Function to calculate probability
const calculateProbability = (data: any[], condition: (item: any) => boolean): number => {
  if (data.length === 0) return 0
  const matchingItems = data.filter(condition)
  return matchingItems.length / data.length
}

// Function to calculate conditional probability
const calculateConditionalProbability = (
  data: any[],
  eventA: (item: any) => boolean,
  eventB: (item: any) => boolean,
): number => {
  if (data.length === 0) return 0
  const itemsB = data.filter(eventB)
  if (itemsB.length === 0) return 0

  const itemsAandB = itemsB.filter(eventA)
  return itemsAandB.length / itemsB.length
}

// Function to generate histogram data
const generateHistogram = (data: number[], bins = 10): { labels: string[]; values: number[] } => {
  if (data.length === 0) return { labels: [], values: [] }

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

// Function to generate normal distribution data
const generateNormalDistribution = (mean: number, stdDev: number, points = 100): { x: number[]; y: number[] } => {
  const x: number[] = []
  const y: number[] = []

  // Generate points from -3 to +3 standard deviations
  const min = mean - 3 * stdDev
  const max = mean + 3 * stdDev
  const step = (max - min) / points

  for (let i = 0; i <= points; i++) {
    const xVal = min + i * step
    const yVal = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((xVal - mean) / stdDev, 2))
    x.push(xVal)
    y.push(yVal)
  }

  return { x, y }
}

export function MLStatisticsDemo() {
  // Data state
  const [originalData, setOriginalData] = useState<any[]>([])
  const [workingData, setWorkingData] = useState<any[]>([])
  const [features, setFeatures] = useState<string[]>([])
  const [numericalFeatures, setNumericalFeatures] = useState<string[]>([])
  const [categoricalFeatures, setCategoricalFeatures] = useState<string[]>([])
  const [targetFeature, setTargetFeature] = useState<string>("")

  // UI state
  const [activeTab, setActiveTab] = useState<string>("descriptive")
  const [activeFeature, setActiveFeature] = useState<string>("")
  const [secondFeature, setSecondFeature] = useState<string>("")
  const [sampleSize, setSampleSize] = useState<number>(100)
  const [confidenceLevel, setConfidenceLevel] = useState<number>(95)
  const [binCount, setBinCount] = useState<number>(10)
  const [showMean, setShowMean] = useState<boolean>(true)
  const [showMedian, setShowMedian] = useState<boolean>(true)
  const [showNormalCurve, setShowNormalCurve] = useState<boolean>(false)

  // Statistics state
  const [dataStats, setDataStats] = useState<any>({})
  const [correlationMatrix, setCorrelationMatrix] = useState<any[]>([])
  const [probabilityStats, setProbabilityStats] = useState<any>({})
  const [hypothesisResults, setHypothesisResults] = useState<any>({})

  // Define the type for dataset items
  type TitanicDataItem = {
    survived: number
    pclass: number
    sex: string
    age: number
    sibsp: number
    parch: number
    fare: number
    embarked: string
  }

  // Initialize data
  useEffect(() => {
    const dataset = TITANIC_DATASET
    setOriginalData(dataset.data)
    setWorkingData(dataset.data)

    // Separate numerical and categorical features
    const numFeatures: string[] = []
    const catFeatures: string[] = []

    dataset.features.forEach((feature) => {
      if (feature in dataset.data[0] && typeof (dataset.data[0] as TitanicDataItem)[feature as keyof TitanicDataItem] === "number") {
        numFeatures.push(feature)
      } else {
        catFeatures.push(feature)
      }
    })

    setFeatures(dataset.features)
    setNumericalFeatures(numFeatures)
    setCategoricalFeatures(catFeatures)
    setTargetFeature(dataset.target)
    setActiveFeature(numFeatures[0] || dataset.features[0])
    setSecondFeature(numFeatures[1] || numFeatures[0] || dataset.features[0])

    // Calculate initial statistics
    calculateDataStats(dataset.data)
    calculateCorrelationMatrix(dataset.data, numFeatures)
    calculateProbabilityStats(dataset.data)
  }, [])

  // Update data when sample size changes
  useEffect(() => {
    if (originalData.length === 0) return

    // Sample the data
    const sampledData = originalData.slice(0, sampleSize)
    setWorkingData(sampledData)

    // Recalculate statistics
    calculateDataStats(sampledData)
    calculateCorrelationMatrix(sampledData, numericalFeatures)
    calculateProbabilityStats(sampledData)
    performHypothesisTest(sampledData)
  }, [originalData, sampleSize, confidenceLevel])

  // Calculate descriptive statistics for the dataset
  const calculateDataStats = (data: any[]) => {
    const stats: any = {
      rowCount: data.length,
      featureCount: features.length,
      features: {},
      target: {},
    }

    // Calculate statistics for numerical features
    numericalFeatures.forEach((feature) => {
      const values = data.map((row) => row[feature]).filter((val) => val !== null && val !== undefined)

      if (values.length === 0) {
        stats.features[feature] = { empty: true }
        return
      }

      const mean = calculateMean(values)
      const median = calculateMedian(values)
      const mode = calculateMode(values)
      const stdDev = calculateStdDev(values, mean)
      const variance = calculateVariance(values, mean)
      const min = Math.min(...values)
      const max = Math.max(...values)
      const q1 = calculateQuantile(values, 0.25)
      const q3 = calculateQuantile(values, 0.75)
      const iqr = q3 - q1

      stats.features[feature] = {
        type: "numerical",
        count: values.length,
        mean,
        median,
        mode,
        stdDev,
        variance,
        min,
        max,
        q1,
        q3,
        iqr,
        histogram: generateHistogram(values, binCount),
      }
    })

    // Calculate statistics for categorical features
    categoricalFeatures.forEach((feature) => {
      const values = data.map((row) => row[feature]).filter((val) => val !== null && val !== undefined)

      if (values.length === 0) {
        stats.features[feature] = { empty: true }
        return
      }

      const valueCounts: Record<string, number> = {}
      values.forEach((val) => {
        valueCounts[val] = (valueCounts[val] || 0) + 1
      })

      const uniqueValues = Object.keys(valueCounts)
      const mostCommon = Object.entries(valueCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([val, count]) => ({ value: val, count }))

      stats.features[feature] = {
        type: "categorical",
        count: values.length,
        uniqueValues: uniqueValues.length,
        mostCommon,
        valueCounts,
      }
    })

    // Target variable statistics
    if (targetFeature) {
      const targetValues = data.map((row) => row[targetFeature]).filter((val) => val !== null && val !== undefined)

      if (typeof targetValues[0] === "number") {
        const mean = calculateMean(targetValues as number[])
        const stdDev = calculateStdDev(targetValues as number[], mean)

        stats.target = {
          type: "numerical",
          mean,
          stdDev,
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

  // Calculate correlation matrix
  const calculateCorrelationMatrix = (data: any[], numFeatures: string[]) => {
    const matrix: any[] = []

    numFeatures.forEach((feature1) => {
      const row: any = { feature: feature1 }

      numFeatures.forEach((feature2) => {
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

  // Calculate probability statistics
  const calculateProbabilityStats = (data: any[]) => {
    const stats: any = {
      survival: {},
      conditional: {},
      bayes: {},
    }

    // Basic probability of survival
    const survivalProb = calculateProbability(data, (item) => item.survived === 1)
    stats.survival.overall = survivalProb

    // Conditional probabilities
    // P(Survived | Female)
    const femaleConditional = calculateConditionalProbability(
      data,
      (item) => item.survived === 1,
      (item) => item.sex === "female",
    )
    stats.conditional.femaleSurvival = femaleConditional

    // P(Survived | Male)
    const maleConditional = calculateConditionalProbability(
      data,
      (item) => item.survived === 1,
      (item) => item.sex === "male",
    )
    stats.conditional.maleSurvival = maleConditional

    // P(Survived | Class = 1)
    const class1Conditional = calculateConditionalProbability(
      data,
      (item) => item.survived === 1,
      (item) => item.pclass === 1,
    )
    stats.conditional.class1Survival = class1Conditional

    // P(Survived | Class = 2)
    const class2Conditional = calculateConditionalProbability(
      data,
      (item) => item.survived === 1,
      (item) => item.pclass === 2,
    )
    stats.conditional.class2Survival = class2Conditional

    // P(Survived | Class = 3)
    const class3Conditional = calculateConditionalProbability(
      data,
      (item) => item.survived === 1,
      (item) => item.pclass === 3,
    )
    stats.conditional.class3Survival = class3Conditional

    // Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
    // P(Female | Survived) = P(Survived | Female) * P(Female) / P(Survived)
    const pFemale = calculateProbability(data, (item) => item.sex === "female")
    const pFemaleSurvived = (femaleConditional * pFemale) / survivalProb
    stats.bayes.femaleSurvived = pFemaleSurvived

    // P(Class=1 | Survived) = P(Survived | Class=1) * P(Class=1) / P(Survived)
    const pClass1 = calculateProbability(data, (item) => item.pclass === 1)
    const pClass1Survived = (class1Conditional * pClass1) / survivalProb
    stats.bayes.class1Survived = pClass1Survived

    setProbabilityStats(stats)
  }

  // Perform hypothesis test
  const performHypothesisTest = (data: any[]) => {
    const results: any = {
      tTest: {},
      zTest: {},
    }

    // T-test for age difference between survivors and non-survivors
    const survivorAges = data
      .filter((item) => item.survived === 1 && item.age !== null && item.age !== undefined)
      .map((item) => item.age)

    const nonSurvivorAges = data
      .filter((item) => item.survived === 0 && item.age !== null && item.age !== undefined)
      .map((item) => item.age)

    if (survivorAges.length > 0 && nonSurvivorAges.length > 0) {
      const meanSurvivorAge = calculateMean(survivorAges)
      const meanNonSurvivorAge = calculateMean(nonSurvivorAges)
      const stdDevSurvivorAge = calculateStdDev(survivorAges, meanSurvivorAge)
      const stdDevNonSurvivorAge = calculateStdDev(nonSurvivorAges, meanNonSurvivorAge)

      // Calculate t-statistic
      const n1 = survivorAges.length
      const n2 = nonSurvivorAges.length
      const pooledStdDev = Math.sqrt(
        ((n1 - 1) * Math.pow(stdDevSurvivorAge, 2) + (n2 - 1) * Math.pow(stdDevNonSurvivorAge, 2)) / (n1 + n2 - 2),
      )

      const tStatistic = (meanSurvivorAge - meanNonSurvivorAge) / (pooledStdDev * Math.sqrt(1 / n1 + 1 / n2))

      // Simplified p-value calculation (not exact but illustrative)
      const degreesOfFreedom = n1 + n2 - 2
      const pValue = 2 * (1 - Math.min(0.9999, Math.abs(tStatistic) / Math.sqrt(degreesOfFreedom)))

      results.tTest = {
        meanSurvivorAge,
        meanNonSurvivorAge,
        difference: meanSurvivorAge - meanNonSurvivorAge,
        tStatistic,
        pValue,
        significant: pValue < 1 - confidenceLevel / 100,
      }
    }

    // Z-test for proportion of survivors by gender
    const females = data.filter((item) => item.sex === "female")
    const males = data.filter((item) => item.sex === "male")

    const femaleSurvivors = females.filter((item) => item.survived === 1)
    const maleSurvivors = males.filter((item) => item.survived === 1)

    if (females.length > 0 && males.length > 0) {
      const pFemale = femaleSurvivors.length / females.length
      const pMale = maleSurvivors.length / males.length

      // Calculate z-statistic
      const pooledProportion = (femaleSurvivors.length + maleSurvivors.length) / (females.length + males.length)
      const standardError = Math.sqrt(
        pooledProportion * (1 - pooledProportion) * (1 / females.length + 1 / males.length),
      )

      const zStatistic = (pFemale - pMale) / standardError

      // Simplified p-value calculation
      const pValue = 2 * (1 - Math.min(0.9999, Math.abs(zStatistic) / 2))

      results.zTest = {
        femaleSurvivalRate: pFemale,
        maleSurvivalRate: pMale,
        difference: pFemale - pMale,
        zStatistic,
        pValue,
        significant: pValue < 1 - confidenceLevel / 100,
      }
    }

    setHypothesisResults(results)
  }

  // Reset to original data
  const handleReset = () => {
    setSampleSize(100)
    setConfidenceLevel(95)
    setBinCount(10)
    setShowMean(true)
    setShowMedian(true)
    setShowNormalCurve(false)
    setWorkingData([...originalData.slice(0, 100)])
    calculateDataStats(originalData.slice(0, 100))
    calculateCorrelationMatrix(originalData.slice(0, 100), numericalFeatures)
    calculateProbabilityStats(originalData.slice(0, 100))
  }

  // Render the descriptive statistics tab
  const renderDescriptiveStatsTab = () => {
    return (
      <div className="space-y-6">
        <Card className="p-4 bg-gradient-to-r from-purple-50 to-indigo-50">
          <h3 className="text-lg font-medium mb-2 text-purple-800">What are Descriptive Statistics?</h3>
          <p className="text-muted-foreground mb-4">
            Descriptive statistics summarize and quantify the main characteristics of a dataset. They help you
            understand the central tendency, spread, and shape of your data distribution.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-3 rounded-md shadow-sm border border-purple-100">
              <div className="flex items-center gap-2 mb-2">
                <Calculator className="h-5 w-5 text-purple-600" />
                <h4 className="text-md font-medium text-purple-800">Central Tendency</h4>
              </div>
              <p className="text-sm text-muted-foreground">Measures that represent the "center" of your data:</p>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>
                  <span className="font-medium">Mean:</span> Average of all values
                </li>
                <li>
                  <span className="font-medium">Median:</span> Middle value (50th percentile)
                </li>
                <li>
                  <span className="font-medium">Mode:</span> Most frequent value
                </li>
              </ul>
            </div>

            <div className="bg-white p-3 rounded-md shadow-sm border border-purple-100">
              <div className="flex items-center gap-2 mb-2">
                <ArrowUpDown className="h-5 w-5 text-purple-600" />
                <h4 className="text-md font-medium text-purple-800">Dispersion</h4>
              </div>
              <p className="text-sm text-muted-foreground">Measures that describe the spread of your data:</p>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>
                  <span className="font-medium">Range:</span> Difference between max and min
                </li>
                <li>
                  <span className="font-medium">Variance:</span> Average squared deviation from mean
                </li>
                <li>
                  <span className="font-medium">Standard Deviation:</span> Square root of variance
                </li>
              </ul>
            </div>

            <div className="bg-white p-3 rounded-md shadow-sm border border-purple-100">
              <div className="flex items-center gap-2 mb-2">
                <Sigma className="h-5 w-5 text-purple-600" />
                <h4 className="text-md font-medium text-purple-800">Distribution</h4>
              </div>
              <p className="text-sm text-muted-foreground">Measures that describe the shape of your data:</p>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>
                  <span className="font-medium">Percentiles:</span> Values that divide data into 100 parts
                </li>
                <li>
                  <span className="font-medium">Quartiles:</span> Q1 (25%), Q2 (50%), Q3 (75%)
                </li>
                <li>
                  <span className="font-medium">IQR:</span> Interquartile range (Q3 - Q1)
                </li>
              </ul>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-medium">Feature Statistics</h3>
            <Select value={activeFeature} onValueChange={setActiveFeature}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select feature" />
              </SelectTrigger>
              <SelectContent>
                {numericalFeatures.map((feature) => (
                  <SelectItem key={feature} value={feature}>
                    {feature}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {activeFeature && dataStats.features && dataStats.features[activeFeature] && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2">
                <div className="aspect-[16/9] bg-muted rounded-md p-4">
                  <Bar
                    data={{
                      labels: dataStats.features[activeFeature].histogram.labels,
                      datasets: [
                        {
                          label: activeFeature,
                          data: dataStats.features[activeFeature].histogram.values,
                          backgroundColor: "rgba(124, 58, 237, 0.5)",
                          borderColor: "rgba(124, 58, 237, 1)",
                          borderWidth: 1,
                        },
                        ...(showNormalCurve
                          ? [
                              {
                                label: "Normal Distribution",
                                type: "line" as const,
                                data: generateNormalDistribution(
                                  dataStats.features[activeFeature].mean,
                                  dataStats.features[activeFeature].stdDev,
                                  dataStats.features[activeFeature].histogram.labels.length,
                                ).y.map(
                                  (y) => y * Math.max(...dataStats.features[activeFeature].histogram.values) * 0.8,
                                ),
                                borderColor: "rgba(220, 38, 38, 1)",
                                borderWidth: 2,
                                tension: 0.4,
                              },
                            ]
                          : []),
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: true,
                      plugins: {
                        legend: {
                          display: true,
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
                </div>

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
                        id="show-normal"
                        checked={showNormalCurve}
                        onCheckedChange={(checked) => setShowNormalCurve(checked as boolean)}
                      />
                      <Label htmlFor="show-normal">Show Normal Distribution</Label>
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                      <Label htmlFor="bin-count">Bins:</Label>
                      <Select value={binCount.toString()} onValueChange={(value) => setBinCount(Number(value))}>
                        <SelectTrigger className="w-[80px]" id="bin-count">
                          <SelectValue placeholder="Bins" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="5">5</SelectItem>
                          <SelectItem value="10">10</SelectItem>
                          <SelectItem value="15">15</SelectItem>
                          <SelectItem value="20">20</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-purple-50 p-2 rounded-md">
                      <div className="text-sm text-muted-foreground">Mean</div>
                      <div className="text-lg font-semibold text-purple-800">
                        {dataStats.features[activeFeature].mean.toFixed(2)}
                      </div>
                    </div>
                    <div className="bg-purple-50 p-2 rounded-md">
                      <div className="text-sm text-muted-foreground">Median</div>
                      <div className="text-lg font-semibold text-purple-800">
                        {dataStats.features[activeFeature].median.toFixed(2)}
                      </div>
                    </div>
                    <div className="bg-purple-50 p-2 rounded-md">
                      <div className="text-sm text-muted-foreground">Std Dev</div>
                      <div className="text-lg font-semibold text-purple-800">
                        {dataStats.features[activeFeature].stdDev.toFixed(2)}
                      </div>
                    </div>
                    <div className="bg-purple-50 p-2 rounded-md">
                      <div className="text-sm text-muted-foreground">Range</div>
                      <div className="text-lg font-semibold text-purple-800">
                        {(dataStats.features[activeFeature].max - dataStats.features[activeFeature].min).toFixed(2)}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h5 className="text-sm font-medium">Five-Number Summary</h5>
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
                          className="absolute h-4 bg-purple-200 border border-purple-400"
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
                          className="absolute h-4 w-0.5 bg-purple-600"
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

                  <div className="bg-purple-50 p-3 rounded-md">
                    <h4 className="text-md font-medium mb-1 text-purple-800">Interpretation</h4>
                    <p className="text-xs text-muted-foreground">
                      {activeFeature === "age" &&
                        "Age distribution shows the spread of passenger ages. The mean age is lower than the median, suggesting some younger passengers (children) pulling the average down."}
                      {activeFeature === "fare" &&
                        "Fare distribution is right-skewed with a few expensive tickets pulling the mean higher than the median. Most passengers paid lower fares, with a few premium tickets."}
                      {activeFeature === "sibsp" &&
                        "Most passengers traveled with few or no siblings/spouses. The distribution is right-skewed with most values at 0 or 1."}
                      {activeFeature === "parch" &&
                        "Most passengers traveled with few or no parents/children. The distribution is right-skewed with most values at 0."}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </Card>

        <Card className="p-4">
          <h3 className="text-lg font-medium mb-4">Categorical Features</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {categoricalFeatures.map((feature) => {
              if (!dataStats.features || !dataStats.features[feature]) return null

              const featureStats = dataStats.features[feature]
              const counts = featureStats.valueCounts || {}
              const labels = Object.keys(counts)
              const values = Object.values(counts)

              return (
                <Card key={feature} className="p-3">
                  <h4 className="text-md font-medium mb-3 text-purple-800">{feature}</h4>
                  <div className="aspect-[16/9] bg-muted rounded-md p-4">
                    <Pie
                      data={{
                        labels,
                        datasets: [
                          {
                            data: values,
                            backgroundColor: [
                              "rgba(124, 58, 237, 0.7)",
                              "rgba(79, 70, 229, 0.7)",
                              "rgba(59, 130, 246, 0.7)",
                              "rgba(16, 185, 129, 0.7)",
                              "rgba(245, 158, 11, 0.7)",
                              "rgba(239, 68, 68, 0.7)",
                            ],
                            borderColor: [
                              "rgba(124, 58, 237, 1)",
                              "rgba(79, 70, 229, 1)",
                              "rgba(59, 130, 246, 1)",
                              "rgba(16, 185, 129, 1)",
                              "rgba(245, 158, 11, 1)",
                              "rgba(239, 68, 68, 1)",
                            ],
                            borderWidth: 1,
                          },
                        ],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {
                          legend: {
                            position: "bottom" as const,
                          },
                          title: {
                            display: true,
                            text: `Distribution of ${feature}`,
                          },
                        },
                      }}
                    />
                  </div>
                  <div className="mt-3">
                    <div className="text-sm text-muted-foreground">
                      {feature === "sex" &&
                        "Gender distribution shows more male passengers than female passengers on the Titanic."}
                      {feature === "embarked" &&
                        "Most passengers embarked from Southampton (S), followed by Cherbourg (C) and Queenstown (Q)."}
                      {feature === "pclass" &&
                        "Passenger class distribution shows more third-class passengers, followed by first and second class."}
                    </div>
                  </div>
                </Card>
              )
            })}
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="text-lg font-medium mb-4">Target Variable: Survival</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2">
              <div className="aspect-[16/9] bg-muted rounded-md p-4">
                <Pie
                  data={{
                    labels: ["Did Not Survive", "Survived"],
                    datasets: [
                      {
                        data: [
                          workingData.filter((item) => item.survived === 0).length,
                          workingData.filter((item) => item.survived === 1).length,
                        ],
                        backgroundColor: ["rgba(239, 68, 68, 0.7)", "rgba(16, 185, 129, 0.7)"],
                        borderColor: ["rgba(239, 68, 68, 1)", "rgba(16, 185, 129, 1)"],
                        borderWidth: 1,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                      legend: {
                        position: "bottom" as const,
                      },
                      title: {
                        display: true,
                        text: "Survival Distribution",
                      },
                    },
                  }}
                />
              </div>
            </div>

            <div>
              <div className="bg-purple-50 p-4 rounded-md">
                <h4 className="text-md font-medium mb-3 text-purple-800">Survival Statistics</h4>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between">
                      <span className="text-sm">Survived:</span>
                      <span className="text-sm font-medium">
                        {workingData.filter((item) => item.survived === 1).length} passengers
                      </span>
                    </div>
                    <Progress
                      value={(workingData.filter((item) => item.survived === 1).length / workingData.length) * 100}
                      className="h-2 mt-1 bg-red-100"
                      style={{ backgroundColor: "rgba(16, 185, 129, 0.2)" }}
                    />
                  </div>

                  <div>
                    <div className="flex justify-between">
                      <span className="text-sm">Did Not Survive:</span>
                      <span className="text-sm font-medium">
                        {workingData.filter((item) => item.survived === 0).length} passengers
                      </span>
                    </div>
                    <Progress
                      value={(workingData.filter((item) => item.survived === 0).length / workingData.length) * 100}
                      className="h-2 mt-1"
                      style={{ backgroundColor: "rgba(239, 68, 68, 0.2)" }}
                    />
                  </div>

                  <div className="pt-2">
                    <div className="text-sm text-muted-foreground">
                      The Titanic dataset shows that approximately{" "}
                      {((workingData.filter((item) => item.survived === 1).length / workingData.length) * 100).toFixed(
                        1,
                      )}
                      % of passengers survived the disaster. This imbalance is important to consider when building
                      predictive models.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>
    )
  }

  // Render the probability tab
  const renderProbabilityTab = () => {
    return (
      <div className="space-y-6">
        <Card className="p-4 bg-gradient-to-r from-blue-50 to-cyan-50">
          <h3 className="text-lg font-medium mb-2 text-blue-800">Probability Basics</h3>
          <p className="text-muted-foreground mb-4">
            Probability is the foundation of machine learning and statistical inference. It measures the likelihood of
            events and helps us quantify uncertainty in our predictions.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-3 rounded-md shadow-sm border border-blue-100">
              <div className="flex items-center gap-2 mb-2">
                <Dice5 className="h-5 w-5 text-blue-600" />
                <h4 className="text-md font-medium text-blue-800">Basic Probability</h4>
              </div>
              <p className="text-sm text-muted-foreground">The probability of an event is a number between 0 and 1:</p>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>P(A) = Number of favorable outcomes / Total number of possible outcomes</li>
                <li>P(not A) = 1 - P(A)</li>
                <li>P(A or B) = P(A) + P(B) - P(A and B)</li>
              </ul>
            </div>

            <div className="bg-white p-3 rounded-md shadow-sm border border-blue-100">
              <div className="flex items-center gap-2 mb-2">
                <ArrowUpDown className="h-5 w-5 text-blue-600" />
                <h4 className="text-md font-medium text-blue-800">Conditional Probability</h4>
              </div>
              <p className="text-sm text-muted-foreground">
                The probability of an event given that another event has occurred:
              </p>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>P(A|B) = P(A and B) / P(B)</li>
                <li>Events are independent if P(A|B) = P(A)</li>
                <li>Chain rule: P(A and B) = P(A) × P(B|A)</li>
              </ul>
            </div>

            <div className="bg-white p-3 rounded-md shadow-sm border border-blue-100">
              <div className="flex items-center gap-2 mb-2">
                <Calculator className="h-5 w-5 text-blue-600" />
                <h4 className="text-md font-medium text-blue-800">Bayes' Theorem</h4>
              </div>
              <p className="text-sm text-muted-foreground">
                A fundamental rule for updating probabilities based on new evidence:
              </p>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>P(A|B) = P(B|A) × P(A) / P(B)</li>
                <li>Posterior = Likelihood × Prior / Evidence</li>
                <li>Foundation of many ML algorithms like Naive Bayes</li>
              </ul>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="text-lg font-medium mb-4">Survival Probabilities</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-md font-medium mb-3 text-blue-800">Basic Probabilities</h4>
              <div className="space-y-4">
                <div className="bg-blue-50 p-3 rounded-md">
                  <div className="flex justify-between">
                    <span className="text-sm">P(Survived)</span>
                    <span className="text-sm font-medium">
                      {probabilityStats.survival?.overall.toFixed(3) || "N/A"}
                    </span>
                  </div>
                  <Progress
                    value={(probabilityStats.survival?.overall || 0) * 100}
                    className="h-2 mt-1"
                    style={{ backgroundColor: "rgba(59, 130, 246, 0.2)" }}
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    The probability of a passenger surviving the Titanic disaster was approximately{" "}
                    {((probabilityStats.survival?.overall || 0) * 100).toFixed(1)}%.
                  </p>
                </div>

                <div className="bg-blue-50 p-3 rounded-md">
                  <div className="flex justify-between">
                    <span className="text-sm">P(Not Survived)</span>
                    <span className="text-sm font-medium">
                      {(1 - (probabilityStats.survival?.overall || 0)).toFixed(3)}
                    </span>
                  </div>
                  <Progress
                    value={(1 - (probabilityStats.survival?.overall || 0)) * 100}
                    className="h-2 mt-1"
                    style={{ backgroundColor: "rgba(59, 130, 246, 0.2)" }}
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    The probability of a passenger not surviving was approximately{" "}
                    {((1 - (probabilityStats.survival?.overall || 0)) * 100).toFixed(1)}%.
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-md font-medium mb-3 text-blue-800">Conditional Probabilities</h4>
              <div className="space-y-4">
                <div className="bg-blue-50 p-3 rounded-md">
                  <div className="flex justify-between">
                    <span className="text-sm">P(Survived | Female)</span>
                    <span className="text-sm font-medium">
                      {probabilityStats.conditional?.femaleSurvival.toFixed(3) || "N/A"}
                    </span>
                  </div>
                  <Progress
                    value={(probabilityStats.conditional?.femaleSurvival || 0) * 100}
                    className="h-2 mt-1"
                    style={{ backgroundColor: "rgba(59, 130, 246, 0.2)" }}
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    The probability of survival given that the passenger was female was approximately{" "}
                    {((probabilityStats.conditional?.femaleSurvival || 0) * 100).toFixed(1)}%.
                  </p>
                </div>

                <div className="bg-blue-50 p-3 rounded-md">
                  <div className="flex justify-between">
                    <span className="text-sm">P(Survived | Male)</span>
                    <span className="text-sm font-medium">
                      {probabilityStats.conditional?.maleSurvival.toFixed(3) || "N/A"}
                    </span>
                  </div>
                  <Progress
                    value={(probabilityStats.conditional?.maleSurvival || 0) * 100}
                    className="h-2 mt-1"
                    style={{ backgroundColor: "rgba(59, 130, 246, 0.2)" }}
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    The probability of survival given that the passenger was male was approximately{" "}
                    {((probabilityStats.conditional?.maleSurvival || 0) * 100).toFixed(1)}%.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-md font-medium mb-3 text-blue-800">Survival by Passenger Class</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-3 rounded-md">
                <div className="flex justify-between">
                  <span className="text-sm">P(Survived | 1st Class)</span>
                  <span className="text-sm font-medium">
                    {probabilityStats.conditional?.class1Survival.toFixed(3) || "N/A"}
                  </span>
                </div>
                <Progress
                  value={(probabilityStats.conditional?.class1Survival || 0) * 100}
                  className="h-2 mt-1"
                  style={{ backgroundColor: "rgba(59, 130, 246, 0.2)" }}
                />
              </div>

              <div className="bg-blue-50 p-3 rounded-md">
                <div className="flex justify-between">
                  <span className="text-sm">P(Survived | 2nd Class)</span>
                  <span className="text-sm font-medium">
                    {probabilityStats.conditional?.class2Survival.toFixed(3) || "N/A"}
                  </span>
                </div>
                <Progress
                  value={(probabilityStats.conditional?.class2Survival || 0) * 100}
                  className="h-2 mt-1"
                  style={{ backgroundColor: "rgba(59, 130, 246, 0.2)" }}
                />
              </div>

              <div className="bg-blue-50 p-3 rounded-md">
                <div className="flex justify-between">
                  <span className="text-sm">P(Survived | 3rd Class)</span>
                  <span className="text-sm font-medium">
                    {probabilityStats.conditional?.class3Survival.toFixed(3) || "N/A"}
                  </span>
                </div>
                <Progress
                  value={(probabilityStats.conditional?.class3Survival || 0) * 100}
                  className="h-2 mt-1"
                  style={{ backgroundColor: "rgba(59, 130, 246, 0.2)" }}
                />
              </div>
            </div>

            <div className="mt-4 bg-white p-3 rounded-md border border-blue-100">
              <h5 className="text-sm font-medium mb-2 text-blue-800">Interpretation</h5>
              <p className="text-sm text-muted-foreground">
                The conditional probabilities reveal important patterns in the Titanic disaster:
              </p>
              <ul className="list-disc pl-5 mt-2 text-sm text-muted-foreground">
                <li>
                  Women had a much higher survival rate (
                  {((probabilityStats.conditional?.femaleSurvival || 0) * 100).toFixed(1)}%) than men (
                  {((probabilityStats.conditional?.maleSurvival || 0) * 100).toFixed(1)}%), reflecting the "women and
                  children first" policy.
                </li>
                <li>
                  First-class passengers had the highest survival rate (
                  {((probabilityStats.conditional?.class1Survival || 0) * 100).toFixed(1)}%), followed by second-class (
                  {((probabilityStats.conditional?.class2Survival || 0) * 100).toFixed(1)}%) and third-class (
                  {((probabilityStats.conditional?.class3Survival || 0) * 100).toFixed(1)}%).
                </li>
                <li>
                  These conditional probabilities suggest that both gender and passenger class were strong predictors of
                  survival.
                </li>
              </ul>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="text-lg font-medium mb-4">Bayes' Theorem Application</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-blue-50 p-4 rounded-md">
              <h4 className="text-md font-medium mb-3 text-blue-800">P(Female | Survived)</h4>
              <div className="text-3xl font-bold text-blue-800 mb-2">
                {probabilityStats.bayes?.femaleSurvived.toFixed(3) || "N/A"}
              </div>
              <p className="text-sm text-muted-foreground">
                Using Bayes' Theorem, we can calculate the probability that a passenger was female, given that they
                survived.
              </p>
              <div className="mt-3 p-3 bg-white rounded-md">
                <p className="text-sm">P(Female | Survived) = P(Survived | Female) × P(Female) / P(Survived)</p>
                <p className="text-sm mt-2">
                  = {probabilityStats.conditional?.femaleSurvival.toFixed(3) || "?"} ×{" "}
                  {(workingData.filter((item) => item.sex === "female").length / workingData.length).toFixed(3)} /{" "}
                  {probabilityStats.survival?.overall.toFixed(3) || "?"}
                </p>
                <p className="text-sm mt-2">= {probabilityStats.bayes?.femaleSurvived.toFixed(3) || "?"}</p>
              </div>
            </div>

            <div className="bg-blue-50 p-4 rounded-md">
              <h4 className="text-md font-medium mb-3 text-blue-800">P(1st Class | Survived)</h4>
              <div className="text-3xl font-bold text-blue-800 mb-2">
                {probabilityStats.bayes?.class1Survived.toFixed(3) || "N/A"}
              </div>
              <p className="text-sm text-muted-foreground">
                Using Bayes' Theorem, we can calculate the probability that a passenger was in first class, given that
                they survived.
              </p>
              <div className="mt-3 p-3 bg-white rounded-md">
                <p className="text-sm">
                  P(1st Class | Survived) = P(Survived | 1st Class) × P(1st Class) / P(Survived)
                </p>
                <p className="text-sm mt-2">
                  = {probabilityStats.conditional?.class1Survival.toFixed(3) || "?"} ×{" "}
                  {(workingData.filter((item) => item.pclass === 1).length / workingData.length).toFixed(3)} /{" "}
                  {probabilityStats.survival?.overall.toFixed(3) || "?"}
                </p>
                <p className="text-sm mt-2">= {probabilityStats.bayes?.class1Survived.toFixed(3) || "?"}</p>
              </div>
            </div>
          </div>

          <div className="mt-6 bg-white p-4 rounded-md border border-blue-100">
            <h4 className="text-md font-medium mb-3 text-blue-800">Why Bayes' Theorem Matters in Machine Learning</h4>
            <p className="text-sm text-muted-foreground">
              Bayes' Theorem is fundamental to many machine learning algorithms and concepts:
            </p>
            <ul className="list-disc pl-5 mt-2 text-sm text-muted-foreground">
              <li>
                <span className="font-medium">Naive Bayes Classifiers:</span> Use Bayes' theorem to predict class
                membership probabilities
              </li>
              <li>
                <span className="font-medium">Bayesian Networks:</span> Model conditional dependencies between random
                variables
              </li>
              <li>
                <span className="font-medium">Bayesian Optimization:</span> Efficiently optimize hyperparameters in ML
                models
              </li>
              <li>
                <span className="font-medium">Bayesian Inference:</span> Update beliefs based on new evidence, crucial
                for online learning
              </li>
            </ul>
            <p className="text-sm text-muted-foreground mt-3">
              In the Titanic example, we can use Bayes' theorem to answer questions like "What's the probability that a
              survivor was female?" This is different from asking "What's the probability that a female survived?" The
              distinction is crucial in many real-world ML applications.
            </p>
          </div>
        </Card>
      </div>
    )
  }

  // Render the distributions tab
  const renderDistributionsTab = () => {
    return (
      <div className="space-y-6">
        <Card className="p-4 bg-gradient-to-r from-green-50 to-emerald-50">
          <h3 className="text-lg font-medium mb-2 text-green-800">Probability Distributions</h3>
          <p className="text-muted-foreground mb-4">
            Probability distributions describe the likelihood of different outcomes in a random experiment.
            Understanding distributions helps in modeling data and making statistical inferences.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white p-3 rounded-md shadow-sm border border-green-100">
              <div className="flex items-center gap-2 mb-2">
                <Bell className="h-5 w-5 text-green-600" />
                <h4 className="text-md font-medium text-green-800">Normal Distribution</h4>
              </div>
              <p className="text-sm text-muted-foreground">The most important continuous distribution in statistics:</p>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>Bell-shaped, symmetric around the mean</li>
                <li>Defined by mean (μ) and standard deviation (σ)</li>
                <li>68-95-99.7 rule: 68% of data within 1σ, 95% within 2σ, 99.7% within 3σ</li>
                <li>Foundation for many statistical methods</li>
              </ul>
            </div>

            <div className="bg-white p-3 rounded-md shadow-sm border border-green-100">
              <div className="flex items-center gap-2 mb-2">
                <BarChart className="h-5 w-5 text-green-600" />
                <h4 className="text-md font-medium text-green-800">Other Common Distributions</h4>
              </div>
              <p className="text-sm text-muted-foreground">Important distributions in machine learning:</p>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>
                  <span className="font-medium">Binomial:</span> Models number of successes in fixed trials
                </li>
                <li>
                  <span className="font-medium">Poisson:</span> Models rare events in a fixed interval
                </li>
                <li>
                  <span className="font-medium">Uniform:</span> Equal probability for all outcomes
                </li>
                <li>
                  <span className="font-medium">Exponential:</span> Models time between events
                </li>
              </ul>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="text-lg font-medium mb-4">Normal Distribution Visualization</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2">
              <div className="aspect-[16/9] bg-muted rounded-md p-4">
                {activeFeature && dataStats.features && dataStats.features[activeFeature] && (
                  <Line
                    data={{
                      labels: Array.from({ length: 100 }, (_, i) => {
                        const min =
                          dataStats.features[activeFeature].mean - 3 * dataStats.features[activeFeature].stdDev
                        const max =
                          dataStats.features[activeFeature].mean + 3 * dataStats.features[activeFeature].stdDev
                        return min + (i / 99) * (max - min)
                      }),
                      datasets: [
                        {
                          label: "Normal Distribution",
                          data: generateNormalDistribution(
                            dataStats.features[activeFeature].mean,
                            dataStats.features[activeFeature].stdDev,
                          ).y,
                          borderColor: "rgba(16, 185, 129, 1)",
                          backgroundColor: "rgba(16, 185, 129, 0.2)",
                          borderWidth: 2,
                          fill: true,
                          tension: 0.4,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: true,
                      plugins: {
                        legend: {
                          display: true,
                        },
                        title: {
                          display: true,
                          text: `Normal Distribution for ${activeFeature}`,
                        },
                        tooltip: {
                          callbacks: {
                            title: (context) => `Value: ${Number.parseFloat(context[0].label).toFixed(2)}`,
                            label: (context) => `Density: ${context.raw}`,
                          },
                        },
                      },
                      scales: {
                        y: {
                          beginAtZero: true,
                          title: {
                            display: true,
                            text: "Probability Density",
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
                )}
              </div>

              <div className="mt-4 flex justify-between">
                <div>
                  <Select value={activeFeature} onValueChange={setActiveFeature}>
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Select feature" />
                    </SelectTrigger>
                    <SelectContent>
                      {numericalFeatures.map((feature) => (
                        <SelectItem key={feature} value={feature}>
                          {feature}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            <div>
              <div className="space-y-4">
                <div className="bg-green-50 p-3 rounded-md">
                  <h4 className="text-md font-medium mb-2 text-green-800">Distribution Parameters</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Mean (μ)</span>
                      <span className="text-sm font-medium">
                        {dataStats.features && dataStats.features[activeFeature]
                          ? dataStats.features[activeFeature].mean.toFixed(2)
                          : "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Std Dev (σ)</span>
                      <span className="text-sm font-medium">
                        {dataStats.features && dataStats.features[activeFeature]
                          ? dataStats.features[activeFeature].stdDev.toFixed(2)
                          : "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Variance (σ²)</span>
                      <span className="text-sm font-medium">
                        {dataStats.features && dataStats.features[activeFeature]
                          ? dataStats.features[activeFeature].variance.toFixed(2)
                          : "N/A"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-green-50 p-3 rounded-md">
                  <h4 className="text-md font-medium mb-2 text-green-800">68-95-99.7 Rule</h4>
                  <div className="space-y-2">
                    <div>
                      <div className="flex justify-between">
                        <span className="text-sm">μ ± 1σ (68%)</span>
                        <span className="text-sm font-medium">
                          {dataStats.features && dataStats.features[activeFeature]
                            ? `${(dataStats.features[activeFeature].mean - dataStats.features[activeFeature].stdDev).toFixed(1)} - ${(dataStats.features[activeFeature].mean + dataStats.features[activeFeature].stdDev).toFixed(1)}`
                            : "N/A"}
                        </span>
                      </div>
                      <Progress value={68} className="h-2 mt-1 bg-green-100" />
                    </div>
                    <div>
                      <div className="flex justify-between">
                        <span className="text-sm">μ ± 2σ (95%)</span>
                        <span className="text-sm font-medium">
                          {dataStats.features && dataStats.features[activeFeature]
                            ? `${(dataStats.features[activeFeature].mean - 2 * dataStats.features[activeFeature].stdDev).toFixed(1)} - ${(dataStats.features[activeFeature].mean + 2 * dataStats.features[activeFeature].stdDev).toFixed(1)}`
                            : "N/A"}
                        </span>
                      </div>
                      <Progress value={95} className="h-2 mt-1 bg-green-100" />
                    </div>
                    <div>
                      <div className="flex justify-between">
                        <span className="text-sm">μ ± 3σ (99.7%)</span>
                        <span className="text-sm font-medium">
                          {dataStats.features && dataStats.features[activeFeature]
                            ? `${(dataStats.features[activeFeature].mean - 3 * dataStats.features[activeFeature].stdDev).toFixed(1)} - ${(dataStats.features[activeFeature].mean + 3 * dataStats.features[activeFeature].stdDev).toFixed(1)}`
                            : "N/A"}
                        </span>
                      </div>
                      <Progress value={99.7} className="h-2 mt-1 bg-green-100" />
                    </div>
                  </div>
                </div>

                <div className="bg-green-50 p-3 rounded-md">
                  <h4 className="text-md font-medium mb-2 text-green-800">Interpretation</h4>
                  <p className="text-sm text-muted-foreground">
                    {activeFeature === "age" &&
                      "The age distribution approximates a normal distribution with some right skew. Most passengers were young to middle-aged adults, with fewer elderly passengers."}
                    {activeFeature === "fare" &&
                      "The fare distribution is heavily right-skewed and does not follow a normal distribution. Most passengers paid lower fares, with a few very expensive tickets."}
                    {activeFeature === "sibsp" &&
                      "The number of siblings/spouses doesn't follow a normal distribution. It's a discrete count variable with most passengers having 0 or 1 siblings/spouses."}
                    {activeFeature === "parch" &&
                      "The number of parents/children doesn't follow a normal distribution. It's a discrete count variable with most passengers having 0 parents/children."}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <h3 className="text-lg font-medium mb-4">Distribution Comparison</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-green-50 p-4 rounded-md">
              <h4 className="text-md font-medium mb-3 text-green-800">Age Distribution by Survival</h4>
              <div className="aspect-[16/9] bg-white rounded-md p-4">
                <Bar
                  data={{
                    labels: ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80"],
                    datasets: [
                      {
                        label: "Survived",
                        data: [
                          workingData.filter(
                            (item) => item.survived === 1 && item.age !== null && item.age >= 0 && item.age <= 10,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.age !== null && item.age >= 11 && item.age <= 20,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.age !== null && item.age >= 21 && item.age <= 30,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.age !== null && item.age >= 31 && item.age <= 40,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.age !== null && item.age >= 41 && item.age <= 50,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.age !== null && item.age >= 51 && item.age <= 60,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.age !== null && item.age >= 61 && item.age <= 70,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.age !== null && item.age >= 71 && item.age <= 80,
                          ).length,
                        ],
                        backgroundColor: "rgba(16, 185, 129, 0.7)",
                        borderColor: "rgba(16, 185, 129, 1)",
                        borderWidth: 1,
                      },
                      {
                        label: "Did Not Survive",
                        data: [
                          workingData.filter(
                            (item) => item.survived === 0 && item.age !== null && item.age >= 0 && item.age <= 10,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.age !== null && item.age >= 11 && item.age <= 20,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.age !== null && item.age >= 21 && item.age <= 30,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.age !== null && item.age >= 31 && item.age <= 40,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.age !== null && item.age >= 41 && item.age <= 50,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.age !== null && item.age >= 51 && item.age <= 60,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.age !== null && item.age >= 61 && item.age <= 70,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.age !== null && item.age >= 71 && item.age <= 80,
                          ).length,
                        ],
                        backgroundColor: "rgba(239, 68, 68, 0.7)",
                        borderColor: "rgba(239, 68, 68, 1)",
                        borderWidth: 1,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                      legend: {
                        position: "top" as const,
                      },
                      title: {
                        display: true,
                        text: "Age Distribution by Survival",
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
                          text: "Age Group",
                        },
                      },
                    },
                  }}
                />
              </div>
            </div>

            <div className="bg-green-50 p-4 rounded-md">
              <h4 className="text-md font-medium mb-3 text-green-800">Fare Distribution by Survival</h4>
              <div className="aspect-[16/9] bg-white rounded-md p-4">
                <Bar
                  data={{
                    labels: ["0-10", "11-20", "21-30", "31-50", "51-100", "100+"],
                    datasets: [
                      {
                        label: "Survived",
                        data: [
                          workingData.filter(
                            (item) => item.survived === 1 && item.fare !== null && item.fare >= 0 && item.fare <= 10,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.fare !== null && item.fare > 10 && item.fare <= 20,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.fare !== null && item.fare > 20 && item.fare <= 30,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.fare !== null && item.fare > 30 && item.fare <= 50,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 1 && item.fare !== null && item.fare > 50 && item.fare <= 100,
                          ).length,
                          workingData.filter((item) => item.survived === 1 && item.fare !== null && item.fare > 100)
                            .length,
                        ],
                        backgroundColor: "rgba(16, 185, 129, 0.7)",
                        borderColor: "rgba(16, 185, 129, 1)",
                        borderWidth: 1,
                      },
                      {
                        label: "Did Not Survive",
                        data: [
                          workingData.filter(
                            (item) => item.survived === 0 && item.fare !== null && item.fare >= 0 && item.fare <= 10,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.fare !== null && item.fare > 10 && item.fare <= 20,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.fare !== null && item.fare > 20 && item.fare <= 30,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.fare !== null && item.fare > 30 && item.fare <= 50,
                          ).length,
                          workingData.filter(
                            (item) => item.survived === 0 && item.fare !== null && item.fare > 50 && item.fare <= 100,
                          ).length,
                          workingData.filter((item) => item.survived === 0 && item.fare !== null && item.fare > 100)
                            .length,
                        ],
                        backgroundColor: "rgba(239, 68, 68, 0.7)",
                        borderColor: "rgba(239, 68, 68, 1)",
                        borderWidth: 1,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                      legend: {
                        position: "top" as const,
                      },
                      title: {
                        display: true,
                        text: "Fare Distribution by Survival",
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
                          text: "Fare (£)",
                        },
                      },
                    },
                  }}
                />
              </div>
            </div>
          </div>

          <div className="mt-6 bg-white p-4 rounded-md border border-green-100">
            <h4 className="text-md font-medium mb-3 text-green-800">Distribution Insights</h4>
            <p className="text-sm text-muted-foreground">
              Understanding the distributions in the Titanic dataset reveals important patterns:
            </p>
            <ul className="list-disc pl-5 mt-2 text-sm text-muted-foreground">
              <li>
                <span className="font-medium">Age Distribution:</span> Children had higher survival rates, while
                middle-aged men had the lowest survival rates.
              </li>
              <li>
                <span className="font-medium">Fare Distribution:</span> Passengers who paid higher fares (typically
                first-class) had much better survival odds.
              </li>
              <li>
                <span className="font-medium">Normal Distribution Approximation:</span> While age roughly follows a
                normal distribution, fare is heavily skewed.
              </li>
              <li>
                <span className="font-medium">Machine Learning Implications:</span> These distribution insights can
                guide feature engineering and model selection.
              </li>
            </ul>
          </div>
        </Card>
      </div>
    )
  }

  // Render the hypothesis testing tab
  const renderHypothesisTab = () => {
    return (
      <div className="space-y-6">
        <Card className="p-4 bg-gradient-to-r from-amber-50 to-yellow-50">
          <h3 className="text-lg font-medium mb-2 text-amber-800">Hypothesis Testing</h3>
          <p className="text-muted-foreground mb-4">
            Hypothesis testing is a statistical method for making decisions based on data. It helps determine if
            observed differences are statistically significant or just due to random chance.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white p-3 rounded-md shadow-sm border border-amber-100">
              <div className="flex items-center gap-2 mb-2">
                <Calculator className="h-5 w-5 text-amber-600" />
                <h4 className="text-md font-medium text-amber-800">Key Concepts</h4>
              </div>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>
                  <span className="font-medium">Null Hypothesis (H₀):</span> Assumes no effect or difference
                </li>
                <li>
                  <span className="font-medium">Alternative Hypothesis (H₁):</span> Assumes there is an effect
                </li>
                <li>
                  <span className="font-medium">p-value:</span> Probability of observing results if H₀ is true
                </li>
                <li>
                  <span className="font-medium">Significance Level (α):</span> Threshold for rejecting H₀ (typically
                  0.05)
                </li>
                <li>
                  <span className="font-medium">Type I Error:</span> Rejecting H₀ when it's true (false positive)
                </li>
                <li>
                  <span className="font-medium">Type II Error:</span> Not rejecting H₀ when it's false (false negative)
                </li>
              </ul>
            </div>

            <div className="bg-white p-3 rounded-md shadow-sm border border-amber-100">
              <div className="flex items-center gap-2 mb-2">
                <ArrowUpDown className="h-5 w-5 text-amber-600" />
                <h4 className="text-md font-medium text-amber-800">Common Tests</h4>
              </div>
              <ul className="list-disc pl-5 mt-2 text-sm">
                <li>
                  <span className="font-medium">t-test:</span> Compares means of two groups (small samples)
                </li>
                <li>
                  <span className="font-medium">z-test:</span> Compares means or proportions (large samples)
                </li>
                <li>
                  <span className="font-medium">ANOVA:</span> Compares means of three or more groups
                </li>
                <li>
                  <span className="font-medium">Chi-square test:</span> Tests relationships between categorical
                  variables
                </li>
                <li>
                  <span className="font-medium">F-test:</span> Compares variances or tests multiple coefficients
                </li>
              </ul>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-medium">Confidence Level</h3>
            <Select value={confidenceLevel.toString()} onValueChange={(value) => setConfidenceLevel(Number(value))}>
              <SelectTrigger className="w-[120px]">
                <SelectValue placeholder="Confidence" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="90">90%</SelectItem>
                <SelectItem value="95">95%</SelectItem>
                <SelectItem value="99">99%</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-amber-50 p-4 rounded-md">
              <h4 className="text-md font-medium mb-3 text-amber-800">t-Test: Age Difference by Survival</h4>
              <div className="space-y-4">
                <div>
                  <h5 className="text-sm font-medium">Hypotheses</h5>
                  <p className="text-sm text-muted-foreground mt-1">
                    <span className="font-medium">H₀:</span> There is no difference in mean age between survivors and
                    non-survivors.
                    <br />
                    <span className="font-medium">H₁:</span> There is a difference in mean age between survivors and
                    non-survivors.
                  </p>
                </div>

                <div>
                  <h5 className="text-sm font-medium">Results</h5>
                  <div className="mt-2 space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Mean Age (Survived)</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.tTest?.meanSurvivorAge?.toFixed(2) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Mean Age (Not Survived)</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.tTest?.meanNonSurvivorAge?.toFixed(2) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Difference</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.tTest?.difference?.toFixed(2) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">t-statistic</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.tTest?.tStatistic?.toFixed(2) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">p-value</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.tTest?.pValue?.toFixed(3) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Significant at {confidenceLevel}% level</span>
                      <span
                        className={`text-sm font-medium ${hypothesisResults.tTest?.significant ? "text-green-600" : "text-red-600"}`}
                      >
                        {hypothesisResults.tTest?.significant ? "Yes" : "No"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-white p-3 rounded-md">
                  <h5 className="text-sm font-medium mb-1">Interpretation</h5>
                  <p className="text-xs text-muted-foreground">
                    {hypothesisResults.tTest?.significant
                      ? `At the ${confidenceLevel}% confidence level, we reject the null hypothesis. There is a statistically significant difference in the mean age between survivors and non-survivors.`
                      : `At the ${confidenceLevel}% confidence level, we fail to reject the null hypothesis. There is not enough evidence to conclude that there is a difference in mean age between survivors and non-survivors.`}
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">
                    {hypothesisResults.tTest?.difference && hypothesisResults.tTest.difference < 0
                      ? `On average, survivors were ${Math.abs(hypothesisResults.tTest.difference).toFixed(1)} years younger than non-survivors.`
                      : hypothesisResults.tTest?.difference && hypothesisResults.tTest.difference > 0
                        ? `On average, survivors were ${hypothesisResults.tTest.difference.toFixed(1)} years older than non-survivors.`
                        : ""}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-amber-50 p-4 rounded-md">
              <h4 className="text-md font-medium mb-3 text-amber-800">z-Test: Survival Rate by Gender</h4>
              <div className="space-y-4">
                <div>
                  <h5 className="text-sm font-medium">Hypotheses</h5>
                  <p className="text-sm text-muted-foreground mt-1">
                    <span className="font-medium">H₀:</span> There is no difference in survival rates between males and
                    females.
                    <br />
                    <span className="font-medium">H₁:</span> There is a difference in survival rates between males and
                    females.
                  </p>
                </div>

                <div>
                  <h5 className="text-sm font-medium">Results</h5>
                  <div className="mt-2 space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Female Survival Rate</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.zTest?.femaleSurvivalRate?.toFixed(2) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Male Survival Rate</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.zTest?.maleSurvivalRate?.toFixed(2) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Difference</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.zTest?.difference?.toFixed(2) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">z-statistic</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.zTest?.zStatistic?.toFixed(2) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">p-value</span>
                      <span className="text-sm font-medium">
                        {hypothesisResults.zTest?.pValue?.toFixed(3) || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Significant at {confidenceLevel}% level</span>
                      <span
                        className={`text-sm font-medium ${hypothesisResults.zTest?.significant ? "text-green-600" : "text-red-600"}`}
                      >
                        {hypothesisResults.zTest?.significant ? "Yes" : "No"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-white p-3 rounded-md">
                  <h5 className="text-sm font-medium mb-1">Interpretation</h5>
                  <p className="text-xs text-muted-foreground">
                    {hypothesisResults.zTest?.significant
                      ? `At the ${confidenceLevel}% confidence level, we reject the null hypothesis. There is a statistically significant difference in survival rates between males and females.`
                      : `At the ${confidenceLevel}% confidence level, we fail to reject the null hypothesis. There is not enough evidence to conclude that there is a difference in survival rates between males and females.`}
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">
                    {hypothesisResults.zTest?.difference && hypothesisResults.zTest.difference > 0
                      ? `The female survival rate was ${(hypothesisResults.zTest.difference * 100).toFixed(1)} percentage points higher than the male survival rate.`
                      : hypothesisResults.zTest?.difference && hypothesisResults.zTest.difference < 0
                        ? `The male survival rate was ${Math.abs(hypothesisResults.zTest.difference * 100).toFixed(1)} percentage points higher than the female survival rate.`
                        : ""}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 bg-white p-4 rounded-md border border-amber-100">
            <h4 className="text-md font-medium mb-3 text-amber-800">
              Why Hypothesis Testing Matters in Machine Learning
            </h4>
            <p className="text-sm text-muted-foreground">
              Hypothesis testing plays several important roles in machine learning:
            </p>
            <ul className="list-disc pl-5 mt-2 text-sm text-muted-foreground">
              <li>
                <span className="font-medium">Feature Selection:</span> Identify statistically significant features that
                have a real relationship with the target variable
              </li>
              <li>
                <span className="font-medium">Model Validation:</span> Determine if a model's performance is
                significantly better than a baseline or another model
              </li>
              <li>
                <span className="font-medium">A/B Testing:</span> Evaluate if changes to a model or system lead to
                statistically significant improvements
              </li>
              <li>
                <span className="font-medium">Outlier Detection:</span> Test if unusual observations are statistically
                different from the rest of the data
              </li>
            </ul>
            <p className="text-sm text-muted-foreground mt-3">
              In the Titanic example, hypothesis testing confirms that gender was a statistically significant factor in
              survival, which would be an important feature to include in a predictive model.
            </p>
          </div>
        </Card>
      </div>
    )
  }

  // Render the sample size controls
  const renderSampleSizeControls = () => {
    return (
      <Card className="p-4">
        <h3 className="text-lg font-medium mb-4">Sample Size Controls</h3>
        <p className="text-muted-foreground mb-4">
          Adjust the sample size to see how it affects statistical measures and hypothesis test results.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <Label htmlFor="sample-size" className="mb-2 block">
              Sample Size: {sampleSize} passengers
            </Label>
            <Select value={sampleSize.toString()} onValueChange={(value) => setSampleSize(Number(value))}>
              <SelectTrigger id="sample-size">
                <SelectValue placeholder="Select sample size" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="20">20 passengers</SelectItem>
                <SelectItem value="50">50 passengers</SelectItem>
                <SelectItem value="80">80 passengers</SelectItem>
                <SelectItem value="100">100 passengers (full dataset)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-2">
              A larger sample size generally leads to more reliable statistics and more powerful hypothesis tests.
            </p>
          </div>

          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 p-4 rounded-md">
            <h4 className="text-sm font-medium mb-2 text-purple-800">Sample Size Effects</h4>
            <ul className="list-disc pl-5 text-sm text-muted-foreground">
              <li>
                <span className="font-medium">Small samples (n=20):</span> High variability, less reliable statistics,
                lower statistical power
              </li>
              <li>
                <span className="font-medium">Medium samples (n=50):</span> Moderate reliability, clearer patterns
                emerge
              </li>
              <li>
                <span className="font-medium">Large samples (n=80+):</span> More stable statistics, higher confidence in
                results
              </li>
              <li>
                <span className="font-medium">Full dataset (n=100):</span> Most reliable results for this particular
                dataset
              </li>
            </ul>
          </div>
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-6 py-4">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-purple-900">Machine Learning Statistics</h2>
          <p className="text-muted-foreground">Interactive statistics demo with the Titanic dataset</p>
        </div>
      </div>

      {renderSampleSizeControls()}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4">
          <TabsTrigger
            value="descriptive"
            className="data-[state=active]:bg-purple-100 data-[state=active]:text-purple-900"
          >
            <Calculator className="h-4 w-4 mr-2" />
            Descriptive Stats
          </TabsTrigger>
          <TabsTrigger
            value="probability"
            className="data-[state=active]:bg-blue-100 data-[state=active]:text-blue-900"
          >
            <Dice5 className="h-4 w-4 mr-2" />
            Probability
          </TabsTrigger>
          <TabsTrigger
            value="distributions"
            className="data-[state=active]:bg-green-100 data-[state=active]:text-green-900"
          >
            <Bell className="h-4 w-4 mr-2" />
            Distributions
          </TabsTrigger>
          <TabsTrigger
            value="hypothesis"
            className="data-[state=active]:bg-amber-100 data-[state=active]:text-amber-900"
          >
            <Sigma className="h-4 w-4 mr-2" />
            Hypothesis Testing
          </TabsTrigger>
        </TabsList>

        <TabsContent value="descriptive">{renderDescriptiveStatsTab()}</TabsContent>
        <TabsContent value="probability">{renderProbabilityTab()}</TabsContent>
        <TabsContent value="distributions">{renderDistributionsTab()}</TabsContent>
        <TabsContent value="hypothesis">{renderHypothesisTab()}</TabsContent>
      </Tabs>
    </div>
  )
}
