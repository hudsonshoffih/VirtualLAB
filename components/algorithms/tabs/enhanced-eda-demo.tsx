"use client"

import React from "react"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from "recharts"
import { Download, RefreshCw, Info, AlertCircle, CheckCircle, Filter, Zap, Database } from "lucide-react"
import { ChartContainer } from "@/components/ui/chart"

// Sample datasets
const datasets = [
  {
    id: "tips",
    name: "Restaurant Tips",
    description: "Dataset containing restaurant tips and related information",
  },
  {
    id: "iris",
    name: "Iris Flower Dataset",
    description: "Classic dataset containing measurements of iris flowers",
  },
  {
    id: "diamonds",
    name: "Diamonds Dataset",
    description: "Dataset containing prices and attributes of diamonds",
  },
]

// Sample data for visualizations
const generateTipsData = () => {
  return [
    { day: "Thur", total_bill: 17.42, tip: 3.68, size: 3, time: "Lunch", sex: "Female" },
    { day: "Fri", total_bill: 19.82, tip: 3.18, size: 2, time: "Lunch", sex: "Male" },
    { day: "Sat", total_bill: 25.28, tip: 5.23, size: 4, time: "Dinner", sex: "Male" },
    { day: "Sun", total_bill: 24.55, tip: 3.83, size: 2, time: "Dinner", sex: "Female" },
    { day: "Thur", total_bill: 14.31, tip: 4.0, size: 2, time: "Lunch", sex: "Female" },
    { day: "Fri", total_bill: 28.17, tip: 6.5, size: 3, time: "Dinner", sex: "Female" },
    { day: "Sat", total_bill: 22.75, tip: 3.25, size: 2, time: "Dinner", sex: "Female" },
    { day: "Sun", total_bill: 20.29, tip: 2.75, size: 2, time: "Dinner", sex: "Female" },
    { day: "Thur", total_bill: 15.77, tip: 2.23, size: 2, time: "Lunch", sex: "Female" },
    { day: "Fri", total_bill: 26.88, tip: 3.12, size: 4, time: "Lunch", sex: "Male" },
    { day: "Sat", total_bill: 25.28, tip: 4.71, size: 4, time: "Dinner", sex: "Male" },
    { day: "Sun", total_bill: 22.76, tip: 3.0, size: 2, time: "Dinner", sex: "Male" },
    { day: "Thur", total_bill: 16.43, tip: 2.3, size: 2, time: "Lunch", sex: "Female" },
    { day: "Fri", total_bill: 18.24, tip: 3.76, size: 2, time: "Lunch", sex: "Male" },
    { day: "Sat", total_bill: 24.06, tip: 3.5, size: 3, time: "Dinner", sex: "Male" },
    { day: "Sun", total_bill: 16.99, tip: 3.5, size: 3, time: "Dinner", sex: "Female" },
  ]
}

const generateIrisData = () => {
  return [
    { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2, species: "setosa" },
    { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2, species: "setosa" },
    { sepal_length: 7.0, sepal_width: 3.2, petal_length: 4.7, petal_width: 1.4, species: "versicolor" },
    { sepal_length: 6.4, sepal_width: 3.2, petal_length: 4.5, petal_width: 1.5, species: "versicolor" },
    { sepal_length: 6.3, sepal_width: 3.3, petal_length: 6.0, petal_width: 2.5, species: "virginica" },
    { sepal_length: 5.8, sepal_width: 2.7, petal_length: 5.1, petal_width: 1.9, species: "virginica" },
    { sepal_length: 5.4, sepal_width: 3.9, petal_length: 1.7, petal_width: 0.4, species: "setosa" },
    { sepal_length: 6.1, sepal_width: 2.8, petal_length: 4.0, petal_width: 1.3, species: "versicolor" },
    { sepal_length: 6.5, sepal_width: 3.0, petal_length: 5.2, petal_width: 2.0, species: "virginica" },
    { sepal_length: 5.0, sepal_width: 3.4, petal_length: 1.5, petal_width: 0.2, species: "setosa" },
    { sepal_length: 5.5, sepal_width: 2.4, petal_length: 3.8, petal_width: 1.1, species: "versicolor" },
    { sepal_length: 7.7, sepal_width: 3.8, petal_length: 6.7, petal_width: 2.2, species: "virginica" },
  ]
}

const generateDiamondsData = () => {
  return [
    { carat: 0.23, cut: "Ideal", color: "E", clarity: "SI2", depth: 61.5, price: 326 },
    { carat: 0.21, cut: "Premium", color: "E", clarity: "SI1", depth: 59.8, price: 326 },
    { carat: 0.23, cut: "Good", color: "E", clarity: "VS1", depth: 56.9, price: 327 },
    { carat: 0.29, cut: "Premium", color: "I", clarity: "VS2", depth: 62.4, price: 334 },
    { carat: 0.31, cut: "Good", color: "J", clarity: "SI2", depth: 63.3, price: 335 },
    { carat: 0.24, cut: "Very Good", color: "J", clarity: "VVS2", depth: 62.8, price: 336 },
    { carat: 0.7, cut: "Ideal", color: "E", clarity: "VS2", depth: 61.7, price: 2757 },
    { carat: 0.86, cut: "Premium", color: "H", clarity: "SI2", depth: 61.0, price: 2757 },
    { carat: 0.75, cut: "Ideal", color: "D", clarity: "SI2", depth: 62.2, price: 2757 },
    { carat: 0.69, cut: "Very Good", color: "F", clarity: "VS1", depth: 62.8, price: 2757 },
  ]
}

// Sample correlation data
const generateCorrelationData = (dataset: any[]) => {
  if (!dataset || dataset.length === 0) return []

  const numericColumns = Object.keys(dataset[0]).filter((key) => typeof dataset[0][key] === "number")

  const correlationMatrix: any[] = []

  numericColumns.forEach((col1) => {
    const row: any = { name: col1 }

    numericColumns.forEach((col2) => {
      // Simulate correlation coefficient between -1 and 1
      // In a real app, you would calculate actual correlation
      if (col1 === col2) {
        row[col2] = 1
      } else {
        // Generate a consistent correlation value for each pair
        const seed = (col1 + col2).split("").reduce((a, b) => a + b.charCodeAt(0), 0)
        const correlation = Math.sin(seed) * 0.8 // Value between -0.8 and 0.8
        row[col2] = Number.parseFloat(correlation.toFixed(2))
      }
    })

    correlationMatrix.push(row)
  })

  return correlationMatrix
}

// Generate summary statistics
const calculateSummaryStats = (data: any[], column: string) => {
  if (!data || data.length === 0) return null

  const values = data.map((d) => d[column]).filter((v) => typeof v === "number" && !isNaN(v))
  if (values.length === 0) return null

  values.sort((a, b) => a - b)

  const sum = values.reduce((acc, val) => acc + val, 0)
  const mean = sum / values.length

  const squaredDiffs = values.map((val) => Math.pow(val - mean, 2))
  const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / values.length
  const stdDev = Math.sqrt(variance)

  const min = values[0]
  const max = values[values.length - 1]
  const q1 = values[Math.floor(values.length * 0.25)]
  const median =
    values.length % 2 === 0
      ? (values[values.length / 2 - 1] + values[values.length / 2]) / 2
      : values[Math.floor(values.length / 2)]
  const q3 = values[Math.floor(values.length * 0.75)]

  return {
    count: values.length,
    mean: Number.parseFloat(mean.toFixed(2)),
    std: Number.parseFloat(stdDev.toFixed(2)),
    min: Number.parseFloat(min.toFixed(2)),
    q1: Number.parseFloat(q1.toFixed(2)),
    median: Number.parseFloat(median.toFixed(2)),
    q3: Number.parseFloat(q3.toFixed(2)),
    max: Number.parseFloat(max.toFixed(2)),
  }
}

// Calculate missing values
const calculateMissingValues = (data: any[]) => {
  if (!data || data.length === 0) return []

  const columns = Object.keys(data[0])
  const result = []

  for (const col of columns) {
    const totalCount = data.length
    const missingCount = data.filter((d) => d[col] === null || d[col] === undefined).length
    const missingPercentage = (missingCount / totalCount) * 100

    result.push({
      column: col,
      missing: missingCount,
      percentage: Number.parseFloat(missingPercentage.toFixed(2)),
    })
  }

  return result
}

// Generate distribution data
const generateDistributionData = (data: any[], column: string, bins = 10) => {
  if (!data || data.length === 0) return []

  const values = data.map((d) => d[column]).filter((v) => typeof v === "number" && !isNaN(v))
  if (values.length === 0) return []

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min
  const binWidth = range / bins

  const histogramData = Array(bins)
    .fill(0)
    .map((_, i) => ({
      binStart: min + i * binWidth,
      binEnd: min + (i + 1) * binWidth,
      count: 0,
      values: [] as number[],
    }))

  values.forEach((value) => {
    const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1)
    histogramData[binIndex].count++
    histogramData[binIndex].values.push(value)
  })

  return histogramData.map((bin) => ({
    bin: `${bin.binStart.toFixed(1)}-${bin.binEnd.toFixed(1)}`,
    count: bin.count,
    frequency: bin.count / values.length,
  }))
}

// Generate categorical distribution
const generateCategoricalDistribution = (data: any[], column: string) => {
  if (!data || data.length === 0) return []

  const counts: Record<string, number> = {}

  data.forEach((d) => {
    const value = String(d[column])
    counts[value] = (counts[value] || 0) + 1
  })

  return Object.entries(counts).map(([category, count]) => ({
    category,
    count,
    percentage: Number.parseFloat(((count / data.length) * 100).toFixed(2)),
  }))
}

// COLORS
const COLORS = [
  "#8884d8",
  "#83a6ed",
  "#8dd1e1",
  "#82ca9d",
  "#a4de6c",
  "#d0ed57",
  "#ffc658",
  "#ff8042",
  "#ff6361",
  "#bc5090",
]

export function EnhancedEdaDemo() {
  const [selectedDataset, setSelectedDataset] = useState("tips")
  const [data, setData] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("overview")
  const [selectedFeature, setSelectedFeature] = useState<string>("")
  const [selectedFeatureType, setSelectedFeatureType] = useState<"numeric" | "categorical">("numeric")
  const [selectedVisualization, setSelectedVisualization] = useState("histogram")
  const [correlationMatrix, setCorrelationMatrix] = useState<any[]>([])
  const [summaryStats, setSummaryStats] = useState<any>(null)
  const [distributionData, setDistributionData] = useState<any[]>([])
  const [missingValues, setMissingValues] = useState<any[]>([])
  const [showOutliers, setShowOutliers] = useState(true)
  const [binCount, setBinCount] = useState(10)
  const [scatterX, setScatterX] = useState<string>("")
  const [scatterY, setScatterY] = useState<string>("")
  const [colorBy, setColorBy] = useState<string>("")

  // Load dataset
  useEffect(() => {
    setLoading(true)

    // Simulate API call to fetch dataset
    setTimeout(() => {
      let newData: any[] = []

      if (selectedDataset === "tips") {
        newData = generateTipsData()
        setSelectedFeature("total_bill")
        setScatterX("total_bill")
        setScatterY("tip")
        setColorBy("time")
      } else if (selectedDataset === "iris") {
        newData = generateIrisData()
        setSelectedFeature("sepal_length")
        setScatterX("sepal_length")
        setScatterY("petal_length")
        setColorBy("species")
      } else if (selectedDataset === "diamonds") {
        newData = generateDiamondsData()
        setSelectedFeature("price")
        setScatterX("carat")
        setScatterY("price")
        setColorBy("cut")
      }

      setData(newData)

      // Generate correlation matrix
      const corrMatrix = generateCorrelationData(newData)
      setCorrelationMatrix(corrMatrix)

      // Calculate missing values
      const missing = calculateMissingValues(newData)
      setMissingValues(missing)

      setLoading(false)
    }, 800)
  }, [selectedDataset])

  // Update summary stats and distribution when feature changes
  useEffect(() => {
    if (!selectedFeature || data.length === 0) return

    // Check if the selected feature is numeric or categorical
    const firstValue = data[0][selectedFeature]
    const isNumeric = typeof firstValue === "number"
    setSelectedFeatureType(isNumeric ? "numeric" : "categorical")

    // Calculate summary statistics for numeric features
    if (isNumeric) {
      const stats = calculateSummaryStats(data, selectedFeature)
      setSummaryStats(stats)

      // Generate distribution data
      const distData = generateDistributionData(data, selectedFeature, binCount)
      setDistributionData(distData)
    } else {
      // Generate categorical distribution
      const catDist = generateCategoricalDistribution(data, selectedFeature)
      setDistributionData(catDist)
      setSummaryStats(null)
    }
  }, [selectedFeature, data, binCount])

  // Get available features from data
  const getFeatures = () => {
    if (!data || data.length === 0) return []
    return Object.keys(data[0])
  }

  // Get numeric features only
  const getNumericFeatures = () => {
    if (!data || data.length === 0) return []
    return Object.keys(data[0]).filter((key) => typeof data[0][key] === "number")
  }

  // Get categorical features only
  const getCategoricalFeatures = () => {
    if (!data || data.length === 0) return []
    return Object.keys(data[0]).filter((key) => typeof data[0][key] !== "number")
  }

  // Render loading state
  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-10 w-48" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-1">
            <Card>
              <CardHeader>
                <Skeleton className="h-6 w-32" />
              </CardHeader>
              <CardContent className="space-y-4">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-5/6" />
                <Skeleton className="h-4 w-full" />
              </CardContent>
            </Card>
          </div>

          <div className="md:col-span-2">
            <Card>
              <CardHeader>
                <Skeleton className="h-6 w-48" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-[300px] w-full" />
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold">Exploratory Data Analysis</h2>
          <p className="text-muted-foreground">
            Analyze and visualize datasets to understand patterns and relationships
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Select value={selectedDataset} onValueChange={setSelectedDataset}>
            <SelectTrigger className="w-[220px]">
              <Database className="mr-2 h-4 w-4" />
              <SelectValue placeholder="Select dataset" />
            </SelectTrigger>
            <SelectContent>
              {datasets.map((dataset) => (
                <SelectItem key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="icon">
                  <RefreshCw className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Refresh data</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
      
      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>Dataset Information</AlertTitle>
        <AlertDescription>
          {selectedDataset === "tips" && "Restaurant tips dataset with information about bills, tips, and customer details."}
          {selectedDataset === "iris" && "Classic Iris flower dataset with measurements of sepal and petal dimensions for three species."}
          {selectedDataset === "diamonds" && "Diamonds dataset with information about carat, cut, color, clarity, and price."}
        </AlertDescription>
      </Alert>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-4 md:w-[600px]">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="distributions">Distributions</TabsTrigger>
          <TabsTrigger value="relationships">Relationships</TabsTrigger>
          <TabsTrigger value="data-quality">Data Quality</TabsTrigger>
        </TabsList>
        
        {/* OVERVIEW TAB */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="md:col-span-1">
              <CardHeader>
                <CardTitle>Dataset Summary</CardTitle>
                <CardDescription>
                  Basic information about the dataset
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium mb-1">Number of Records</h4>
                  <div className="text-2xl font-bold">{data.length}</div>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium mb-1">Number of Features</h4>
                  <div className="text-2xl font-bold">{getFeatures().length}</div>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium mb-1">Feature Types</h4>
                  <div className="flex gap-2">
                    <Badge variant="outline" className="flex items-center gap-1">
                      <span className="h-2 w-2 rounded-full bg-primary"></span>
                      Numeric: {getNumericFeatures().length}
                    </Badge>
                    <Badge variant="outline" className="flex items-center gap-1">
                      <span className="h-2 w-2 rounded-full bg-orange-500"></span>
                      Categorical: {getCategoricalFeatures().length}
                    </Badge>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium mb-1">Missing Values</h4>
                  <div className="flex items-center gap-2">
                    <span className="text-2xl font-bold">
                      {missingValues.reduce((acc, curr) => acc + curr.missing, 0)}
                    </span>
                    <span className="text-muted-foreground text-sm">
                      ({((missingValues.reduce((acc, curr) => acc + curr.missing, 0) / (data.length * getFeatures().length)) * 100).toFixed(2)}%)
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="md:col-span-2">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Feature Overview</CardTitle>
                  <CardDescription>
                    Summary of available features
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm" className="flex items-center gap-1">
                  <Download className="h-4 w-4" />
                  Export
                </Button>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border">
                  <div className="grid grid-cols-4 bg-muted p-3 rounded-t-md">
                    <div className="font-medium">Feature</div>
                    <div className="font-medium">Type</div>
                    <div className="font-medium">Unique Values</div>
                    <div className="font-medium">Missing</div>
                  </div>
                  <div className="divide-y max-h-[300px] overflow-auto">
                    {getFeatures().map((feature) => {
                      const uniqueValues = new Set(data.map(d => d[feature])).size
                      const missingCount = data.filter(d => d[feature] === null || d[feature] === undefined).length
                      const featureType = typeof data[0][feature] === 'number' ? 'Numeric' : 'Categorical'
                      
                      return (
                        <div key={feature} className="grid grid-cols-4 p-3 hover:bg-muted/50">
                          <div className="font-medium">{feature}</div>
                          <div>
                            <Badge variant={featureType === 'Numeric' ? 'default' : 'secondary'}>
                              {featureType}
                            </Badge>
                          </div>
                          <div>{uniqueValues}</div>
                          <div className="flex items-center gap-2">
                            {missingCount > 0 ? (
                              <>
                                <span className="text-red-500">{missingCount}</span>
                                <span className="text-muted-foreground text-xs">
                                  ({((missingCount / data.length) * 100).toFixed(1)}%)
                                </span>
                              </>
                            ) : (
                              <span className="text-green-500">0</span>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle>Correlation Matrix</CardTitle>
              <CardDescription>
                Explore relationships between numeric features
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] w-full overflow-auto">
                <div className="min-w-[600px] min-h-[400px]">
                  <ResponsiveContainer width="100%" height={400}>
                    <ChartContainer>
                      {correlationMatrix.length > 0 ? (
                        <div className="grid grid-cols-[auto_1fr] gap-4">
                          <div></div>
                          <div className="grid" style={{ 
                            gridTemplateColumns: `repeat(${getNumericFeatures().length}, minmax(80px, 1fr))` 
                          }}>
                            {getNumericFeatures().map((feature) => (
                              <div key={feature} className="px-2 py-1 text-sm font-medium text-center truncate">
                                {feature}
                              </div>
                            ))}
                          </div>
                          
                          {correlationMatrix.map((row, rowIndex) => (
                            <React.Fragment key={rowIndex}>
                              <div className="flex items-center px-2 py-1 text-sm font-medium">
                                {row.name}
                              </div>
                              
                              <div className="grid" style={{ 
                                gridTemplateColumns: `repeat(${getNumericFeatures().length}, minmax(80px, 1fr))` 
                              }}>
                                {getNumericFeatures().map((feature) => {
                                  const value = row[feature]
                                  let color = "bg-gray-200 dark:bg-gray-700"
                                  
                                  if (value === 1) {
                                    color = "bg-primary/90"
                                  } else if (value >= 0.7) {
                                    color = "bg-primary/70"
                                  } else if (value >= 0.4) {
                                    color = "bg-primary/50"
                                  } else if (value >= 0.1) {
                                    color = "bg-primary/30"
                                  } else if (value >= -0.1) {
                                    color = "bg-gray-200 dark:bg-gray-700"
                                  } else if (value >= -0.4) {
                                    color = "bg-red-300/30"
                                  } else if (value >= -0.7) {
                                    color = "bg-red-300/50"
                                  } else {
                                    color = "bg-red-300/70"
                                  }
                                  
                                  return (
                                    <div 
                                      key={feature} 
                                      className={`m-1 p-2 rounded-md text-center ${color} hover:opacity-80 transition-opacity`}
                                      title={`${row.name} vs ${feature}: ${value}`}
                                    >
                                      {value}
                                    </div>
                                  )
                                })}
                              </div>
                            </React.Fragment>
                          ))}
                        </div>
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <p className="text-muted-foreground">No numeric features available for correlation analysis</p>
                        </div>
                      )}
                    </ChartContainer>
                  </ResponsiveContainer>
                </div>
              </div>
              
              <div className="mt-4 text-sm text-muted-foreground">
                <p>
                  <span className="font-medium">Correlation interpretation:</span> Values close to 1 indicate strong positive correlation, 
                  values close to -1 indicate strong negative correlation, and values close to 0 indicate little to no correlation.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* DISTRIBUTIONS TAB */}
        <TabsContent value="distributions" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="md:col-span-1">
              <CardHeader>
                <CardTitle>Feature Selection</CardTitle>
                <CardDescription>
                  Select a feature to analyze its distribution
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="feature-select">Feature</Label>
                  <Select value={selectedFeature} onValueChange={setSelectedFeature}>
                    <SelectTrigger id="feature-select">
                      <SelectValue placeholder="Select feature" />
                    </SelectTrigger>
                    <SelectContent>
                      <div className="mb-2 px-2 text-xs text-muted-foreground font-medium">Numeric Features</div>
                      {getNumericFeatures().map((feature) => (
                        <SelectItem key={feature} value={feature}>
                          {feature}
                        </SelectItem>
                      ))}
                      <Separator className="my-2" />
                      <div className="mb-2 px-2 text-xs text-muted-foreground font-medium">Categorical Features</div>
                      {getCategoricalFeatures().map((feature) => (
                        <SelectItem key={feature} value={feature}>
                          {feature}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                {selectedFeatureType === "numeric" && (
                  <>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label htmlFor="bin-count">Number of Bins</Label>
                        <span className="text-sm text-muted-foreground">{binCount}</span>
                      </div>
                      <Slider 
                        id="bin-count"
                        min={5} 
                        max={20} 
                        step={1} 
                        value={[binCount]} 
                        onValueChange={(value) => setBinCount(value[0])} 
                      />
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Switch 
                        id="show-outliers" 
                        checked={showOutliers} 
                        onCheckedChange={setShowOutliers} 
                      />
                      <Label htmlFor="show-outliers">Show Outliers</Label>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="visualization-type">Visualization Type</Label>
                      <Select value={selectedVisualization} onValueChange={setSelectedVisualization}>
                        <SelectTrigger id="visualization-type">
                          <SelectValue placeholder="Select visualization" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="histogram">Histogram</SelectItem>
                          <SelectItem value="boxplot">Box Plot</SelectItem>
                          <SelectItem value="density">Density Plot</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </>
                )}
                
                {selectedFeatureType === "categorical" && (
                  <div className="space-y-2">
                    <Label htmlFor="visualization-type">Visualization Type</Label>
                    <Select value={selectedVisualization} onValueChange={setSelectedVisualization}>
                      <SelectTrigger id="visualization-type">
                        <SelectValue placeholder="Select visualization" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="bar">Bar Chart</SelectItem>
                        <SelectItem value="pie">Pie Chart</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
                
                {summaryStats && (
                  <div className="space-y-2 pt-4 border-t">
                    <h4 className="font-medium">Summary Statistics</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Count:</span>
                        <span className="font-medium">{summaryStats.count}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Mean:</span>
                        <span className="font-medium">{summaryStats.mean}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Std Dev:</span>
                        <span className="font-medium">{summaryStats.std}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Min:</span>
                        <span className="font-medium">{summaryStats.min}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Q1:</span>
                        <span className="font-medium">{summaryStats.q1}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Median:</span>
                        <span className="font-medium">{summaryStats.median}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Q3:</span>
                        <span className="font-medium">{summaryStats.q3}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Max:</span>
                        <span className="font-medium">{summaryStats.max}</span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
            
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>
                  Distribution of {selectedFeature}
                </CardTitle>
                <CardDescription>
                  {selectedFeatureType === "numeric" 
                    ? "Visualize the distribution of values" 
                    : "Visualize the frequency of categories"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {selectedFeatureType === "numeric" && (
                  <div className="h-[400px]">
                    {selectedVisualization === "histogram" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={distributionData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="bin" />
                          <YAxis />
                          <RechartsTooltip 
                            formatter={(value: any, name: any) => [value, name === "count" ? "Count" : "Frequency"]}
                            labelFormatter={(label) => `Range: ${label}`}
                          />
                          <Bar dataKey="count" fill="#8884d8" name="Count" />
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                    
                    {selectedVisualization === "boxplot" && (
                      <div className="flex items-center justify-center h-full">
                        <div className="w-full max-w-md">
                          <div className="relative h-20">
                            <div className="absolute inset-0 flex items-center">
                              <div className="w-full h-0.5 bg-muted-foreground/30"></div>
                            </div>
                            
                            {/* Min */}
                            <div 
                              className="absolute top-0 bottom-0 flex flex-col items-center justify-center"
                              style={{ left: "0%" }}
                            >
                              <div className="h-full w-0.5 bg-muted-foreground/50"></div>
                              <div className="mt-2 text-xs">{summaryStats.min}</div>
                            </div>
                            
                            {/* Q1 */}
                            <div 
                              className="absolute top-0 bottom-0 flex flex-col items-center justify-center"
                              style={{ 
                                left: `${((summaryStats.q1 - summaryStats.min) / (summaryStats.max - summaryStats.min)) * 100}%` 
                              }}
                            >
                              <div className="h-full w-0.5 bg-primary"></div>
                            </div>
                            
                            {/* Box from Q1 to Q3 */}
                            <div 
                              className="absolute top-1/4 h-1/2 bg-primary/20"
                              style={{ 
                                left: `${((summaryStats.q1 - summaryStats.min) / (summaryStats.max - summaryStats.min)) * 100}%`,
                                width: `${((summaryStats.q3 - summaryStats.q1) / (summaryStats.max - summaryStats.min)) * 100}%`
                              }}
                            ></div>
                            
                            {/* Median */}
                            <div 
                              className="absolute top-1/4 bottom-1/4 flex flex-col items-center justify-center"
                              style={{ 
                                left: `${((summaryStats.median - summaryStats.min) / (summaryStats.max - summaryStats.min)) * 100}%` 
                              }}
                            >
                              <div className="h-full w-1 bg-primary"></div>
                            </div>
                            
                            {/* Q3 */}
                            <div 
                              className="absolute top-0 bottom-0 flex flex-col items-center justify-center"
                              style={{ 
                                left: `${((summaryStats.q3 - summaryStats.min) / (summaryStats.max - summaryStats.min)) * 100}%` 
                              }}
                            >
                              <div className="h-full w-0.5 bg-primary"></div>
                            </div>
                            
                            {/* Max */}
                            <div 
                              className="absolute top-0 bottom-0 flex flex-col items-center justify-center"
                              style={{ left: "100%" }}
                            >
                              <div className="h-full w-0.5 bg-muted-foreground/50"></div>
                              <div className="mt-2 text-xs">{summaryStats.max}</div>
                            </div>
                            
                            {/* Mean indicator */}
                            <div 
                              className="absolute top-0 flex flex-col items-center justify-center"
                              style={{ 
                                left: `${((summaryStats.mean - summaryStats.min) / (summaryStats.max - summaryStats.min)) * 100}%` 
                              }}
                            >
                              <div className="h-5 w-5 rounded-full bg-red-500 flex items-center justify-center text-white text-xs">μ</div>
                              <div className="mt-1 text-xs text-red-500">{summaryStats.mean}</div>
                            </div>
                          </div>
                          
                          <div className="mt-12 text-center text-sm text-muted-foreground">
                            <div className="flex items-center justify-center gap-4">
                              <div className="flex items-center gap-1">
                                <div className="w-3 h-3 bg-primary"></div>
                                <span>IQR (Q1-Q3)</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                <span>Mean</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <div className="w-3 h-3 bg-primary/50"></div>
                                <span>Median</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {selectedVisualization === "density" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={distributionData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="bin" />
                          <YAxis />
                          <RechartsTooltip 
                            formatter={(value: any) => [value, "Frequency"]}
                            labelFormatter={(label) => `Range: ${label}`}
                          />
                          <Area type="monotone" dataKey="frequency" stroke="#8884d8" fill="#8884d8" />
                        </AreaChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                )}
                
                {selectedFeatureType === "categorical" && (
                  <div className="h-[400px]">
                    {selectedVisualization === "bar" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={distributionData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="category" />
                          <YAxis />
                          <RechartsTooltip 
                            formatter={(value: any, name: any) => [value, name === "count" ? "Count" : "Percentage"]}
                            labelFormatter={(label) => `Category: ${label}`}
                          />
                          <Bar dataKey="count" fill="#8884d8" name="Count">
                            {distributionData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                    
                    {selectedVisualization === "pie" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={distributionData}
                            cx="50%"
                            cy="50%"
                            labelLine={true}
                            outerRadius={120}
                            fill="#8884d8"
                            dataKey="count"
                            nameKey="category"
                            label={({ category, percentage }) => `${category}: ${percentage}%`}
                          >
                            {distributionData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <RechartsTooltip 
                            formatter={(value: any, name: any, props: any) => [
                              `Count: ${value} (${props.payload.percentage}%)`, 
                              props.payload.category
                            ]}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* RELATIONSHIPS TAB */}
        <TabsContent value="relationships" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Feature Relationships</CardTitle>
              <CardDescription>
                Explore relationships between different features
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="x-feature">X-Axis Feature</Label>
                  <Select value={scatterX} onValueChange={setScatterX}>
                    <SelectTrigger id="x-feature">
                      <SelectValue placeholder="Select X feature" />
                    </SelectTrigger>
                    <SelectContent>
                      {getNumericFeatures().map((feature) => (
                        <SelectItem key={feature} value={feature}>
                          {feature}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <Label htmlFor="y-feature">Y-Axis Feature</Label>
                  <Select value={scatterY} onValueChange={setScatterY}>
                    <SelectTrigger id="y-feature">
                      <SelectValue placeholder="Select Y feature" />
                    </SelectTrigger>
                    <SelectContent>
                      {getNumericFeatures().map((feature) => (
                        <SelectItem key={feature} value={feature}>
                          {feature}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <Label htmlFor="color-by">Color By</Label>
                  <Select value={colorBy} onValueChange={setColorBy}>
                    <SelectTrigger id="color-by">
                      <SelectValue placeholder="Select category" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">None</SelectItem>
                      {getCategoricalFeatures().map((feature) => (
                        <SelectItem key={feature} value={feature}>
                          {feature}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="h-[500px]">
                {scatterX && scatterY && (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid />
                      <XAxis 
                        type="number" 
                        dataKey={scatterX} 
                        name={scatterX} 
                        label={{ value: scatterX, position: 'bottom' }} 
                      />
                      <YAxis 
                        type="number" 
                        dataKey={scatterY} 
                        name={scatterY} 
                        label={{ value: scatterY, angle: -90, position: 'left' }} 
                      />
                      <RechartsTooltip 
                        cursor={{ strokeDasharray: '3 3' }} 
                        formatter={(value: any, name: any) => [value, name]}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="bg-background border rounded-md shadow-md p-2 text-sm">
                                <p className="font-medium">{colorBy ? `${colorBy}: ${payload[0].payload[colorBy]}` : 'Data Point'}</p>
                                <p>{`${scatterX}: ${payload[0].value}`}</p>
                                <p>{`${scatterY}: ${payload[1].value}`}</p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Legend />
                      
                      {colorBy ? (
                        (() => {
                          // Get unique categories
                          const categories = [...new Set(data.map(d => d[colorBy]))];
                          
                          return categories.map((category, index) => {
                            const filteredData = data.filter(d => d[colorBy] === category);
                            
                            return (
                              <Scatter 
                                key={category} 
                                name={`${category}`} 
                                data={filteredData} 
                                fill={COLORS[index % COLORS.length]} 
                              />
                            );
                          });
                        })()
                      ) : (
                        <Scatter name="All Data" data={data} fill="#8884d8" />
                      )}
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </div>
              
              {scatterX && scatterY && (
                <div className="p-4 bg-muted rounded-md">
                  <h4 className="font-medium mb-2">Relationship Analysis</h4>
                  <p className="text-sm text-muted-foreground">
                    {(() => {
                      if (!data || data.length === 0) return "No data available for analysis.";
                      
                      // Calculate correlation between X and Y
                      const xValues = data.map(d => d[scatterX]);
                      const yValues = data.map(d => d[scatterY]);
                      
                      // Find correlation row for X
                      const corrRow = correlationMatrix.find(row => row.name === scatterX);
                      const correlation = corrRow ? corrRow[scatterY] : null;
                      
                      if (correlation === null) return "Correlation could not be calculated.";
                      
                      let relationshipDesc = "";
                      if (correlation > 0.7) {
                        relationshipDesc = "strong positive";
                      } else if (correlation > 0.3) {
                        relationshipDesc = "moderate positive";
                      } else if (correlation > 0) {
                        relationshipDesc = "weak positive";
                      } else if (correlation > -0.3) {
                        relationshipDesc = "weak negative";
                      } else if (correlation > -0.7) {
                        relationshipDesc = "moderate negative";
                      } else {
                        relationshipDesc = "strong negative";
                      }
                      
                      return `There appears to be a ${relationshipDesc} correlation (${correlation.toFixed(2)}) between ${scatterX} and ${scatterY}.`;
                    })()}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* DATA QUALITY TAB */}
        <TabsContent value="data-quality" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Missing Values</CardTitle>
                <CardDescription>
                  Analyze missing values in the dataset
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {missingValues.some(mv => mv.missing > 0) ? (
                    <>
                      <div className="space-y-3">
                        {missingValues
                          .filter(mv => mv.missing > 0)
                          .sort((a, b) => b.missing - a.missing)
                          .map(mv => (
                            <div key={mv.column} className="space-y-1">
                              <div className="flex justify-between text-sm">
                                <span>{mv.column}</span>
                                <span className="text-muted-foreground">
                                  {mv.missing} ({mv.percentage}%)
                                </span>
                              </div>
                              <Progress value={mv.percentage} className="h-2" />
                            </div>
                          ))}
                      </div>
                      
                      <div className="pt-2 text-sm text-muted-foreground">
                        <p>
                          <AlertCircle className="inline-block h-4 w-4 mr-1" />
                          Missing values may affect analysis results. Consider imputation or filtering.
                        </p>
                      </div>
                    </>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                      <CheckCircle className="h-12 w-12 text-green-500 mb-2" />
                      <h3 className="text-lg font-medium">No Missing Values</h3>
                      <p className="text-muted-foreground">
                        This dataset is complete with no missing values.
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Outlier Detection</CardTitle>
                <CardDescription>
                  Identify potential outliers in numeric features
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {getNumericFeatures().length > 0 ? (
                    <>
                      <div className="space-y-4">
                        {getNumericFeatures().map(feature => {
                          const stats = calculateSummaryStats(data, feature);
                          if (!stats) return null;
                          
                          // Calculate IQR and outlier boundaries
                          const iqr = stats.q3 - stats.q1;
                          const lowerBound = stats.q1 - 1.5 * iqr;
                          const upperBound = stats.q3 + 1.5 * iqr;
                          
                          // Count outliers
                          const outliers = data.filter(d => {
                            const val = d[feature];
                            return typeof val === 'number' && (val < lowerBound || val > upperBound);
                          });
                          
                          const outlierPercentage = (outliers.length / data.length) * 100;
                          
                          return (
                            <div key={feature} className="space-y-1">
                              <div className="flex justify-between text-sm">
                                <span>{feature}</span>
                                <span className="text-muted-foreground">
                                  {outliers.length} outliers ({outlierPercentage.toFixed(1)}%)
                                </span>
                              </div>
                              <div className="relative h-2 bg-muted rounded-full overflow-hidden">
                                <div 
                                  className="absolute h-full bg-primary rounded-full" 
                                  style={{ 
                                    left: `${((lowerBound - stats.min) / (stats.max - stats.min)) * 100}%`,
                                    width: `${((upperBound - lowerBound) / (stats.max - stats.min)) * 100}%`
                                  }}
                                ></div>
                              </div>
                              <div className="flex justify-between text-xs text-muted-foreground">
                                <span>Min: {stats.min}</span>
                                <span>Lower: {lowerBound.toFixed(2)}</span>
                                <span>Upper: {upperBound.toFixed(2)}</span>
                                <span>Max: {stats.max}</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      
                      <div className="pt-2 text-sm text-muted-foreground">
                        <p>
                          <Info className="inline-block h-4 w-4 mr-1" />
                          Outliers are defined as values below Q1-1.5*IQR or above Q3+1.5*IQR.
                        </p>
                      </div>
                    </>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                      <AlertCircle className="h-12 w-12 text-muted-foreground mb-2" />
                      <h3 className="text-lg font-medium">No Numeric Features</h3>
                      <p className="text-muted-foreground">
                        Outlier detection requires numeric features.
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle>Data Quality Report</CardTitle>
              <CardDescription>
                Summary of data quality issues and recommendations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-muted p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-primary/10 rounded-full">
                        <Database className="h-5 w-5 text-primary" />
                      </div>
                      <h3 className="font-medium">Completeness</h3>
                    </div>
                    
                    <div className="space-y-2">
                      <Progress 
                        value={100 - ((missingValues.reduce((acc, curr) => acc + curr.missing, 0) / (data.length * getFeatures().length)) * 100)}
                        className="h-2" 
                      />
                      
                      <div className="flex justify-between text-sm">
                        <span>Missing Values:</span>
                        <span className="font-medium">
                          {missingValues.reduce((acc, curr) => acc + curr.missing, 0)} / {data.length * getFeatures().length}
                        </span>
                      </div>
                      
                      <div className="text-xs text-muted-foreground">
                        {missingValues.some(mv => mv.missing > 0) ? (
                          <p>Some features have missing values that may need to be addressed.</p>
                        ) : (
                          <p>Dataset is complete with no missing values.</p>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-primary/10 rounded-full">
                        <Filter className="h-5 w-5 text-primary" />
                      </div>
                      <h3 className="font-medium">Outliers</h3>
                    </div>
                    
                    <div className="space-y-2">
                      {getNumericFeatures().length > 0 ? (
                        <>
                          <div className="text-sm">
                            <span>Features with outliers:</span>
                            <span className="font-medium ml-1">
                              {getNumericFeatures().filter(feature => {
                                const stats = calculateSummaryStats(data, feature);
                                if (!stats) return false;
                                
                                const iqr = stats.q3 - stats.q1;
                                const lowerBound = stats.q1 - 1.5 * iqr;
                                const upperBound = stats.q3 + 1.5 * iqr;
                                
                                return data.some(d => {
                                  const val = d[feature];
                                  return typeof val === 'number' && (val < lowerBound || val > upperBound);
                                });
                              }).length} / {getNumericFeatures().length}
                            </span>
                          </div>
                          
                          <div className="text-xs text-muted-foreground">
                            <p>Consider transforming or removing outliers for better analysis results.</p>
                          </div>
                        </>
                      ) : (
                        <div className="text-xs text-muted-foreground">
                          <p>No numeric features available for outlier detection.</p>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-primary/10 rounded-full">
                        <Zap className="h-5 w-5 text-primary" />
                      </div>
                      <h3 className="font-medium">Recommendations</h3>
                    </div>
                    
                    <div className="space-y-2 text-xs text-muted-foreground">
                      <ul className="space-y-1 list-disc list-inside">
                        {missingValues.some(mv => mv.missing > 0) && (
                          <li>Consider imputing missing values or removing incomplete records</li>
                        )}
                        
                        {getNumericFeatures().some(feature => {
                          const stats = calculateSummaryStats(data, feature);
                          if (!stats) return false;
                          
                          const iqr = stats.q3 - stats.q1;
                          const lowerBound = stats.q1 - 1.5 * iqr;
                          const upperBound = stats.q3 + 1.5 * iqr;
                          
                          return data.some(d => {
                            const val = d[feature];
                            return typeof val === 'number' && (val < lowerBound || val > upperBound);
                          });
                        }) && (
                          <li>Address outliers through transformation or removal</li>
                        )}
                        
                        <li>Explore feature correlations to identify potential redundancies</li>
                        <li>Consider feature scaling for machine learning applications</li>
                      </ul>
                    </div>
                  </div>
                </div>
                
                <div className="pt-4 border-t">
                  <h3 className="font-medium mb-2">Data Quality Score</h3>
                  <div className="flex items-center gap-4">
                    <div className="w-full h-3 bg-muted rounded-full overflow-hidden">
                      {(() => {
                        // Calculate a simple data quality score
                        const missingPercentage = (missingValues.reduce((acc, curr) => acc + curr.missing, 0) / (data.length * getFeatures().length)) * 100;
                        
                        // Count features with outliers
                        const featuresWithOutliers = getNumericFeatures().filter(feature => {
                          const stats = calculateSummaryStats(data, feature);
                          if (!stats) return false;
                          
                          const iqr = stats.q3 - stats.q1;
                          const lowerBound = stats.q1 - 1.5 * iqr;
                          const upperBound = stats.q3 + 1.5 * iqr;
                          
                          return data.some(d => {
                            const val = d[feature];
                            return typeof val === 'number' && (val < lowerBound || val > upperBound);
                          });
                        }).length;
                        
                        const outlierPercentage = getNumericFeatures().length > 0 
                          ? (featuresWithOutliers / getNumericFeatures().length) * 100 / 3 // Divide by 3 to reduce impact
                          : 0;
                        
                        // Calculate score (100 - penalties)
                        const score = 100 - missingPercentage - outlierPercentage;
                        
                        // Determine color based on score
                        let color = "bg-red-500";
                        if (score >= 90) {
                          color = "bg-green-500";
                        } else if (score >= 70) {
                          color = "bg-yellow-500";
                        } else if (score >= 50) {
                          color = "bg-orange-500";
                        }
                        
                        return (
                          <>
                            <div className={`h-full ${color}`} style={{ width: `${score}%` }}></div>
                            <span className="ml-2 text-sm font-medium">{Math.round(score)}%</span>
                          </>
                        );
                      })()}
                    </div>
                  </div>
                  
                  <div className="mt-4 text-sm text-muted-foreground">
                    <p>
                      This score is based on data completeness, outlier presence, and other quality factors. 
                      Higher scores indicate better quality data for analysis.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

