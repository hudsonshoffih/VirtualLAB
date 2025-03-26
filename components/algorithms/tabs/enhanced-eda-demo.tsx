"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Play, Pause, RefreshCw, BarChart3, LineChart, ScatterChartIcon } from "lucide-react"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js"
import { Chart, Scatter } from "react-chartjs-2"

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler)

// Sample datasets
const datasets = {
  normal: {
    name: "Normal Distribution",
    description: "A symmetric bell-shaped distribution with most values clustered around the mean.",
    generateData: (mean = 50, stdDev = 10, size = 100) => {
      // Box-Muller transform to generate normally distributed random numbers
      const generateGaussian = (mean: number, stdDev: number) => {
        const u1 = Math.random()
        const u2 = Math.random()
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
        return z0 * stdDev + mean
      }

      return Array.from({ length: size }, () => generateGaussian(mean, stdDev))
    },
  },
  rightSkewed: {
    name: "Right-Skewed Distribution",
    description: "A distribution with a long tail to the right, where mean > median.",
    generateData: (mean = 30, skewness = 2, size = 100) => {
      // Generate log-normal distribution (right-skewed)
      const sigma = Math.sqrt(Math.log(1 + skewness))
      const mu = Math.log(mean) - (sigma * sigma) / 2

      return Array.from({ length: size }, () => {
        const z = Math.sqrt(-2.0 * Math.log(Math.random())) * Math.cos(2.0 * Math.PI * Math.random())
        return Math.exp(mu + sigma * z)
      })
    },
  },
  leftSkewed: {
    name: "Left-Skewed Distribution",
    description: "A distribution with a long tail to the left, where mean < median.",
    generateData: (mean = 70, skewness = 2, size = 100) => {
      // Generate left-skewed distribution by reflecting a right-skewed one
      const rightSkewed = datasets.rightSkewed.generateData(30, skewness, size)
      const max = Math.max(...rightSkewed) * 2

      return rightSkewed.map((value) => max - value)
    },
  },
}

// Statistical functions
const calculateMean = (data: number[]): number => {
  return data.reduce((sum, value) => sum + value, 0) / data.length
}

const calculateMedian = (data: number[]): number => {
  const sorted = [...data].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)

  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2
  }

  return sorted[middle]
}

const calculateVariance = (data: number[], mean: number): number => {
  return data.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / data.length
}

const calculateStandardDeviation = (variance: number): number => {
  return Math.sqrt(variance)
}

const calculateMin = (data: number[]): number => {
  return Math.min(...data)
}

const calculateMax = (data: number[]): number => {
  return Math.max(...data)
}

const calculateQuantile = (data: number[], q: number): number => {
  const sorted = [...data].sort((a, b) => a - b)
  const pos = (sorted.length - 1) * q
  const base = Math.floor(pos)
  const rest = pos - base

  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base])
  } else {
    return sorted[base]
  }
}

// Generate histogram data
const generateHistogramData = (data: number[], bins = 10): { labels: string[]; values: number[] } => {
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min
  const binWidth = range / bins

  const histogramValues = Array(bins).fill(0)
  const histogramLabels = Array(bins)
    .fill(0)
    .map((_, i) => {
      const binStart = min + i * binWidth
      const binEnd = min + (i + 1) * binWidth
      return `${binStart.toFixed(1)}-${binEnd.toFixed(1)}`
    })

  data.forEach((value) => {
    // Handle edge case for max value
    if (value === max) {
      histogramValues[bins - 1]++
      return
    }

    const binIndex = Math.floor((value - min) / binWidth)
    histogramValues[binIndex]++
  })

  return { labels: histogramLabels, values: histogramValues }
}

export function EnhancedEdaDemo() {
  // Dataset selection and parameters
  const [selectedDataset, setSelectedDataset] = useState<string>("normal")
  const [datasetSize, setDatasetSize] = useState<number>(100)
  const [binCount, setBinCount] = useState<number>(10)

  // Statistical parameters that user can adjust
  const [userMean, setUserMean] = useState<number>(50)
  const [userStdDev, setUserStdDev] = useState<number>(10)
  const [userSkewness, setUserSkewness] = useState<number>(2)

  // Generated data and calculated statistics
  const [data, setData] = useState<number[]>([])
  const [mean, setMean] = useState<number>(0)
  const [median, setMedian] = useState<number>(0)
  const [variance, setVariance] = useState<number>(0)
  const [stdDev, setStdDev] = useState<number>(0)
  const [min, setMin] = useState<number>(0)
  const [max, setMax] = useState<number>(0)
  const [q1, setQ1] = useState<number>(0)
  const [q3, setQ3] = useState<number>(0)

  // Animation state
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [speed, setSpeed] = useState<number>(1)
  const [activeTab, setActiveTab] = useState<string>("distribution")
  const [showVideo, setShowVideo] = useState<boolean>(false)

  // Generate data based on selected dataset and parameters
  const generateData = () => {
    let newData: number[] = []

    if (selectedDataset === "normal") {
      newData = datasets.normal.generateData(userMean, userStdDev, datasetSize)
    } else if (selectedDataset === "rightSkewed") {
      newData = datasets.rightSkewed.generateData(userMean, userSkewness, datasetSize)
    } else if (selectedDataset === "leftSkewed") {
      newData = datasets.leftSkewed.generateData(userMean, userSkewness, datasetSize)
    }

    setData(newData)

    // Calculate statistics
    const calculatedMean = calculateMean(newData)
    setMean(calculatedMean)

    setMedian(calculateMedian(newData))

    const calculatedVariance = calculateVariance(newData, calculatedMean)
    setVariance(calculatedVariance)

    setStdDev(calculateStandardDeviation(calculatedVariance))
    setMin(calculateMin(newData))
    setMax(calculateMax(newData))
    setQ1(calculateQuantile(newData, 0.25))
    setQ3(calculateQuantile(newData, 0.75))
  }

  // Generate data when parameters change
  useEffect(() => {
    generateData()
  }, [selectedDataset, userMean, userStdDev, userSkewness, datasetSize])

  // Handle animation
  useEffect(() => {
    let animationInterval: NodeJS.Timeout | null = null

    if (isPlaying) {
      if (selectedDataset === "normal") {
        let currentStdDev = userStdDev
        animationInterval = setInterval(() => {
          currentStdDev = currentStdDev >= 20 ? 5 : currentStdDev + 1
          setUserStdDev(currentStdDev)
        }, 1000 / speed)
      } else {
        let currentSkewness = userSkewness
        animationInterval = setInterval(() => {
          currentSkewness = currentSkewness >= 5 ? 1 : currentSkewness + 0.5
          setUserSkewness(currentSkewness)
        }, 1000 / speed)
      }
    }

    return () => {
      if (animationInterval) clearInterval(animationInterval)
    }
  }, [isPlaying, speed, selectedDataset, userStdDev, userSkewness])

  // Prepare chart data
  const histogramData = generateHistogramData(data, binCount)

  const histogramChartData = {
    labels: histogramData.labels,
    datasets: [
      {
        label: "Frequency",
        data: histogramData.values,
        backgroundColor: "rgba(53, 162, 235, 0.5)",
        borderColor: "rgba(53, 162, 235, 1)",
        borderWidth: 1,
      },
    ],
  }

  // Normal distribution curve
  const generateNormalDistributionCurve = () => {
    const points = 100
    const xMin = Math.min(min, mean - 3 * stdDev)
    const xMax = Math.max(max, mean + 3 * stdDev)

    const labels = Array.from({ length: points }, (_, i) => {
      return xMin + (i / (points - 1)) * (xMax - xMin)
    })

    const values = labels.map((x) => {
      const exponent = -Math.pow(x - mean, 2) / (2 * variance)
      return (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent)
    })

    // Scale the values to match the histogram scale
    const maxValue = Math.max(...histogramData.values)
    const maxNormalValue = Math.max(...values)
    const scaledValues = values.map((v) => (v / maxNormalValue) * maxValue)

    return { labels, values: scaledValues }
  }

  const normalDistribution = generateNormalDistributionCurve()

  const distributionChartData = {
    labels: normalDistribution.labels,
    datasets: [
      {
        type: "line" as const,
        label: "Normal Distribution",
        data: normalDistribution.values,
        borderColor: "rgba(255, 99, 132, 1)",
        borderWidth: 2,
        fill: false,
        tension: 0.4,
        yAxisID: "y",
      },
      {
        type: "bar" as const,
        label: "Histogram",
        data: histogramData.values,
        backgroundColor: "rgba(53, 162, 235, 0.5)",
        borderColor: "rgba(53, 162, 235, 1)",
        borderWidth: 1,
        barPercentage: 1,
        categoryPercentage: 1,
        yAxisID: "y",
      },
    ],
  }

  // Scatter plot data
  const scatterPlotData = {
    datasets: [
      {
        label: "Data Points",
        data: data.map((value, index) => ({ x: index, y: value })),
        backgroundColor: "rgba(75, 192, 192, 0.5)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
        pointRadius: 3,
        pointHoverRadius: 5,
      },
      {
        label: "Mean",
        data: data.map((_, index) => ({ x: index, y: mean })),
        backgroundColor: "rgba(255, 99, 132, 0.5)",
        borderColor: "rgba(255, 99, 132, 1)",
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 0,
        showLine: true,
      },
      {
        label: "Median",
        data: data.map((_, index) => ({ x: index, y: median })),
        backgroundColor: "rgba(54, 162, 235, 0.5)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 0,
        showLine: true,
        borderDash: [5, 5],
      },
    ],
  }

  // Box plot data (simplified as a line chart)
  const boxPlotData = {
    labels: ["Box Plot"],
    datasets: [
      {
        label: "Min",
        data: [min],
        backgroundColor: "rgba(75, 192, 192, 0.5)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
      },
      {
        label: "Q1",
        data: [q1],
        backgroundColor: "rgba(54, 162, 235, 0.5)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 1,
      },
      {
        label: "Median",
        data: [median],
        backgroundColor: "rgba(255, 206, 86, 0.5)",
        borderColor: "rgba(255, 206, 86, 1)",
        borderWidth: 1,
      },
      {
        label: "Q3",
        data: [q3],
        backgroundColor: "rgba(153, 102, 255, 0.5)",
        borderColor: "rgba(153, 102, 255, 1)",
        borderWidth: 1,
      },
      {
        label: "Max",
        data: [max],
        backgroundColor: "rgba(255, 159, 64, 0.5)",
        borderColor: "rgba(255, 159, 64, 1)",
        borderWidth: 1,
      },
    ],
  }

  // Reset to default values
  const handleReset = () => {
    setIsPlaying(false)
    setUserMean(50)
    setUserStdDev(10)
    setUserSkewness(2)
    setBinCount(10)
    setDatasetSize(100)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row gap-4 items-start">
        {/* Main visualization area */}
        <Card className="p-6 flex-1">
          <div className="mb-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
              <h3 className="text-xl font-semibold">Statistical Visualization</h3>
              <p className="text-muted-foreground">
                Explore how changing statistical parameters affects data distribution
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="outline" size="icon" onClick={() => setIsPlaying(!isPlaying)}>
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button variant="outline" size="icon" onClick={handleReset}>
                <RefreshCw className="h-4 w-4" />
              </Button>
              <Select value={speed.toString()} onValueChange={(value) => setSpeed(Number(value))}>
                <SelectTrigger className="w-24">
                  <SelectValue placeholder="Speed" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0.5">0.5x</SelectItem>
                  <SelectItem value="1">1x</SelectItem>
                  <SelectItem value="1.5">1.5x</SelectItem>
                  <SelectItem value="2">2x</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="distribution">
                <BarChart3 className="h-4 w-4 mr-2" />
                Distribution
              </TabsTrigger>
              <TabsTrigger value="scatter">
                <ScatterChartIcon className="h-4 w-4 mr-2" />
                Data Points
              </TabsTrigger>
              <TabsTrigger value="statistics">
                <LineChart className="h-4 w-4 mr-2" />
                Statistics
              </TabsTrigger>
            </TabsList>

            <TabsContent value="distribution" className="space-y-4">
              <div className="aspect-[16/9] bg-card rounded-md">
                <Chart
                  type="bar"
                  data={distributionChartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: true,
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
                          text: "Value",
                        },
                      },
                    },
                    plugins: {
                      legend: {
                        position: "top" as const,
                      },
                      title: {
                        display: true,
                        text: `Distribution with Mean=${mean.toFixed(2)}, Median=${median.toFixed(2)}, StdDev=${stdDev.toFixed(2)}`,
                      },
                      tooltip: {
                        callbacks: {
                          title: (context) => `Value: ${context[0].label}`,
                        },
                      },
                    },
                  }}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Number of Bins</label>
                  <div className="flex items-center gap-4">
                    <Slider
                      value={[binCount]}
                      min={5}
                      max={30}
                      step={1}
                      className="flex-1"
                      onValueChange={(value) => setBinCount(value[0])}
                    />
                    <span className="text-sm w-8 text-right">{binCount}</span>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Sample Size</label>
                  <div className="flex items-center gap-4">
                    <Slider
                      value={[datasetSize]}
                      min={20}
                      max={500}
                      step={10}
                      className="flex-1"
                      onValueChange={(value) => setDatasetSize(value[0])}
                    />
                    <span className="text-sm w-8 text-right">{datasetSize}</span>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="scatter" className="space-y-4">
              <div className="aspect-[16/9] bg-card rounded-md">
                <Scatter
                  data={scatterPlotData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                      y: {
                        title: {
                          display: true,
                          text: "Value",
                        },
                      },
                      x: {
                        title: {
                          display: true,
                          text: "Index",
                        },
                      },
                    },
                    plugins: {
                      legend: {
                        position: "top" as const,
                      },
                      title: {
                        display: true,
                        text: `Data Points with Mean and Median Lines`,
                      },
                    },
                  }}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-2">Mean vs Median</h4>
                  <div className="bg-muted p-3 rounded-md">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Mean:</div>
                      <div className="text-right font-medium">{mean.toFixed(2)}</div>
                      <div>Median:</div>
                      <div className="text-right font-medium">{median.toFixed(2)}</div>
                      <div>Difference:</div>
                      <div className="text-right font-medium">{(mean - median).toFixed(2)}</div>
                      <div>Skewness:</div>
                      <div className="text-right font-medium">
                        {mean > median
                          ? "Positive (right-skewed)"
                          : mean < median
                            ? "Negative (left-skewed)"
                            : "Symmetric"}
                      </div>
                    </div>
                  </div>
                </Card>

                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-2">Data Range</h4>
                  <div className="bg-muted p-3 rounded-md">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Minimum:</div>
                      <div className="text-right font-medium">{min.toFixed(2)}</div>
                      <div>Maximum:</div>
                      <div className="text-right font-medium">{max.toFixed(2)}</div>
                      <div>Range:</div>
                      <div className="text-right font-medium">{(max - min).toFixed(2)}</div>
                      <div>IQR:</div>
                      <div className="text-right font-medium">{(q3 - q1).toFixed(2)}</div>
                    </div>
                  </div>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="statistics" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-2">Central Tendency</h4>
                  <div className="bg-muted p-3 rounded-md">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Mean:</div>
                      <div className="text-right font-medium">{mean.toFixed(2)}</div>
                      <div>Median:</div>
                      <div className="text-right font-medium">{median.toFixed(2)}</div>
                      <div>Difference:</div>
                      <div className="text-right font-medium">{(mean - median).toFixed(2)}</div>
                    </div>
                  </div>
                </Card>

                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-2">Dispersion</h4>
                  <div className="bg-muted p-3 rounded-md">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Variance:</div>
                      <div className="text-right font-medium">{variance.toFixed(2)}</div>
                      <div>Standard Deviation:</div>
                      <div className="text-right font-medium">{stdDev.toFixed(2)}</div>
                      <div>Coefficient of Variation:</div>
                      <div className="text-right font-medium">{((stdDev / mean) * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                </Card>

                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-2">Quartiles</h4>
                  <div className="bg-muted p-3 rounded-md">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Q1 (25%):</div>
                      <div className="text-right font-medium">{q1.toFixed(2)}</div>
                      <div>Q2 (50%):</div>
                      <div className="text-right font-medium">{median.toFixed(2)}</div>
                      <div>Q3 (75%):</div>
                      <div className="text-right font-medium">{q3.toFixed(2)}</div>
                      <div>IQR:</div>
                      <div className="text-right font-medium">{(q3 - q1).toFixed(2)}</div>
                    </div>
                  </div>
                </Card>

                <Card className="p-4">
                  <h4 className="text-sm font-medium mb-2">Distribution Shape</h4>
                  <div className="bg-muted p-3 rounded-md">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Skewness:</div>
                      <div className="text-right font-medium">
                        {mean > median
                          ? "Positive (right-skewed)"
                          : mean < median
                            ? "Negative (left-skewed)"
                            : "Symmetric"}
                      </div>
                      <div>Mean - Median:</div>
                      <div className="text-right font-medium">{(mean - median).toFixed(2)}</div>
                    </div>
                  </div>
                </Card>
              </div>

              <div className="aspect-[16/9] bg-card rounded-md flex items-center justify-center">
                <div className="w-full max-w-md p-4">
                  <h4 className="text-center font-medium mb-4">Five-Number Summary</h4>
                  <div className="relative h-16 bg-muted rounded-md">
                    {/* Box plot visualization */}
                    <div className="absolute top-0 left-0 w-full h-4 flex items-center justify-center">
                      <div className="absolute h-0.5 bg-gray-400 w-full"></div>

                      {/* Min */}
                      <div
                        className="absolute h-4 w-0.5 bg-gray-600"
                        style={{ left: `${((min - min) / (max - min)) * 100}%` }}
                      ></div>

                      {/* Q1 */}
                      <div
                        className="absolute h-4 w-0.5 bg-gray-600"
                        style={{ left: `${((q1 - min) / (max - min)) * 100}%` }}
                      ></div>

                      {/* Box from Q1 to Q3 */}
                      <div
                        className="absolute h-4 bg-blue-200 border border-blue-400"
                        style={{
                          left: `${((q1 - min) / (max - min)) * 100}%`,
                          width: `${((q3 - q1) / (max - min)) * 100}%`,
                        }}
                      ></div>

                      {/* Median */}
                      <div
                        className="absolute h-4 w-0.5 bg-blue-600"
                        style={{ left: `${((median - min) / (max - min)) * 100}%` }}
                      ></div>

                      {/* Q3 */}
                      <div
                        className="absolute h-4 w-0.5 bg-gray-600"
                        style={{ left: `${((q3 - min) / (max - min)) * 100}%` }}
                      ></div>

                      {/* Max */}
                      <div
                        className="absolute h-4 w-0.5 bg-gray-600"
                        style={{ left: `${((max - min) / (max - min)) * 100}%` }}
                      ></div>
                    </div>

                    {/* Labels */}
                    <div className="absolute top-6 left-0 w-full flex justify-between text-xs text-gray-500">
                      <div className="transform -translate-x-1/2">Min: {min.toFixed(1)}</div>
                      <div
                        className="transform -translate-x-1/2"
                        style={{ left: `${((q1 - min) / (max - min)) * 100}%` }}
                      >
                        Q1: {q1.toFixed(1)}
                      </div>
                      <div
                        className="transform -translate-x-1/2"
                        style={{ left: `${((median - min) / (max - min)) * 100}%` }}
                      >
                        Median: {median.toFixed(1)}
                      </div>
                      <div
                        className="transform -translate-x-1/2"
                        style={{ left: `${((q3 - min) / (max - min)) * 100}%` }}
                      >
                        Q3: {q3.toFixed(1)}
                      </div>
                      <div className="transform -translate-x-1/2">Max: {max.toFixed(1)}</div>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </Card>

        {/* Controls sidebar */}
        <Card className="p-6 w-full md:w-80">
          <h3 className="font-medium mb-4">Controls</h3>

          <div className="space-y-6">
            <div>
              <label className="text-sm font-medium mb-2 block">Distribution Type</label>
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger>
                  <SelectValue placeholder="Select distribution" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="normal">Normal Distribution</SelectItem>
                  <SelectItem value="rightSkewed">Right-Skewed Distribution</SelectItem>
                  <SelectItem value="leftSkewed">Left-Skewed Distribution</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                {selectedDataset === "normal"
                  ? "A symmetric bell-shaped distribution with most values clustered around the mean."
                  : selectedDataset === "rightSkewed"
                    ? "A distribution with a long tail to the right, where mean > median."
                    : "A distribution with a long tail to the left, where mean < median."}
              </p>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Mean</label>
              <div className="flex items-center gap-4">
                <Slider
                  value={[userMean]}
                  min={10}
                  max={90}
                  step={1}
                  className="flex-1"
                  onValueChange={(value) => setUserMean(value[0])}
                />
                <span className="text-sm w-8 text-right">{userMean}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">Actual calculated mean: {mean.toFixed(2)}</p>
            </div>

            {selectedDataset === "normal" ? (
              <div>
                <label className="text-sm font-medium mb-2 block">Standard Deviation</label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[userStdDev]}
                    min={1}
                    max={20}
                    step={1}
                    className="flex-1"
                    onValueChange={(value) => setUserStdDev(value[0])}
                  />
                  <span className="text-sm w-8 text-right">{userStdDev}</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">Actual calculated std dev: {stdDev.toFixed(2)}</p>
              </div>
            ) : (
              <div>
                <label className="text-sm font-medium mb-2 block">Skewness</label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[userSkewness]}
                    min={1}
                    max={5}
                    step={0.1}
                    className="flex-1"
                    onValueChange={(value) => setUserSkewness(value[0])}
                  />
                  <span className="text-sm w-8 text-right">{userSkewness.toFixed(1)}</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">Mean - Median: {(mean - median).toFixed(2)}</p>
              </div>
            )}

            <div className="pt-4">
              <h4 className="text-sm font-medium mb-2">Summary Statistics</h4>
              <div className="bg-muted p-3 rounded-md">
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>Mean:</div>
                  <div className="text-right font-medium">{mean.toFixed(2)}</div>
                  <div>Median:</div>
                  <div className="text-right font-medium">{median.toFixed(2)}</div>
                  <div>Std Dev:</div>
                  <div className="text-right font-medium">{stdDev.toFixed(2)}</div>
                  <div>Variance:</div>
                  <div className="text-right font-medium">{variance.toFixed(2)}</div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Explanation section */}
      <Card className="p-6">
        <h3 className="font-medium mb-4">Explanation</h3>
        <p className="text-muted-foreground mb-4">
          This demo visualizes how statistical measures like mean, median, standard deviation, and variance relate to
          data distributions. By adjusting the parameters, you can see how these statistics change and affect the shape
          of the distribution.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-sm font-medium mb-2">Key Observations</h4>
            <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
              <li>In a normal distribution, the mean and median are approximately equal</li>
              <li>In a right-skewed distribution, the mean is greater than the median</li>
              <li>In a left-skewed distribution, the mean is less than the median</li>
              <li>Standard deviation controls the spread of the distribution</li>
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-medium mb-2">Tips</h4>
            <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
              <li>Increase the standard deviation to see a wider spread of values</li>
              <li>Compare the mean and median to identify skewness in the data</li>
              <li>Use the animation feature to see how changing parameters affects the distribution</li>
              <li>Switch between distribution types to understand different data patterns</li>
            </ul>
          </div>
        </div>
      </Card>

      {/* Video section */}
      <Card className="p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="font-medium">Video Tutorial</h3>
          <Button variant="outline" size="sm" onClick={() => setShowVideo(!showVideo)}>
            {showVideo ? "Hide Video" : "Show Video"}
          </Button>
        </div>

        {showVideo ? (
          <div className="aspect-video bg-muted rounded-md flex items-center justify-center">
            <p className="text-muted-foreground">
              Add your YouTube video embed here. Replace this placeholder with an iframe element.
            </p>
            {/* Example YouTube embed (uncomment and replace with your video ID)
            <iframe 
              className="w-full h-full rounded-md"
              src="https://www.youtube.com/embed/YOUR_VIDEO_ID" 
              title="Statistical Measures Tutorial"
              frameBorder="0" 
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
              allowFullScreen
            ></iframe>
            */}
          </div>
        ) : (
          <p className="text-muted-foreground">
            Click "Show Video" to view a tutorial on statistical measures and their impact on data distributions.
          </p>
        )}
      </Card>
    </div>
  )
}

