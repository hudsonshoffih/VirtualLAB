"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Card } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Play, Pause, RefreshCw, Download, ChevronDown, ChevronUp } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export function EnhancedEdaDemo() {
  // State for controls
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [datasetType, setDatasetType] = useState("normal")
  const [datasetSize, setDatasetSize] = useState(100)
  const [showAdvancedStats, setShowAdvancedStats] = useState(false)

  // Distribution parameters
  const [parameters, setParameters] = useState({
    mean: 50,
    standardDeviation: 15,
    skewness: 0,
    outliers: 0,
  })

  // Define type for stats
  type Stats = {
    mean: number
    median: number
    mode: number
    standardDeviation: number
    variance: number
    min: number
    max: number
    q1: number
    q3: number
    iqr: number
    skewness: number
    kurtosis: number
  }

  // Statistical measures
  const [stats, setStats] = useState<Stats>({
    mean: 50,
    median: 50,
    mode: 50,
    standardDeviation: 15,
    variance: 225,
    min: 0,
    max: 100,
    q1: 40,
    q3: 60,
    iqr: 20,
    skewness: 0,
    kurtosis: 3,
  })

  // Chart references
  const distributionChartRef = useRef<HTMLCanvasElement>(null)
  const histogramChartRef = useRef<HTMLCanvasElement>(null)
  const boxplotChartRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Generate dataset based on parameters
  const generateDataset = () => {
    const data: number[] = []
    const size = datasetSize

    // Generate data based on distribution type
    switch (datasetType) {
      case "normal":
        // Normal distribution using Box-Muller transform
        for (let i = 0; i < size; i++) {
          let u = 0,
            v = 0
          while (u === 0) u = Math.random()
          while (v === 0) v = Math.random()

          let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)

          // Apply skewness if needed
          if (parameters.skewness !== 0) {
            const delta = parameters.skewness * Math.sign(z) * (Math.exp(Math.abs(parameters.skewness * z)) - 1)
            z = z + delta
          }

          // Convert to desired mean and standard deviation
          const value = z * parameters.standardDeviation + parameters.mean
          data.push(value)
        }
        break

      case "uniform":
        // Uniform distribution
        const range = parameters.standardDeviation * 3.5
        const min = parameters.mean - range / 2
        const max = parameters.mean + range / 2

        for (let i = 0; i < size; i++) {
          const value = min + Math.random() * (max - min)
          data.push(value)
        }
        break

      case "bimodal":
        // Bimodal distribution (mixture of two normals)
        for (let i = 0; i < size; i++) {
          let u = 0,
            v = 0
          while (u === 0) u = Math.random()
          while (v === 0) v = Math.random()

          const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)

          // 50% chance of coming from first or second mode
          const mode = Math.random() < 0.5 ? -1 : 1
          const modeOffset = parameters.standardDeviation * 1.5 * mode

          const value = (z * parameters.standardDeviation) / 1.5 + parameters.mean + modeOffset
          data.push(value)
        }
        break

      case "exponential":
        // Exponential distribution
        const lambda = 1 / parameters.standardDeviation

        for (let i = 0; i < size; i++) {
          const u = Math.random()
          const value = -Math.log(1 - u) / lambda + parameters.mean - parameters.standardDeviation
          data.push(value)
        }
        break
    }

    // Add outliers if specified
    if (parameters.outliers > 0) {
      const outlierCount = Math.floor((size * parameters.outliers) / 100)
      for (let i = 0; i < outlierCount; i++) {
        const index = Math.floor(Math.random() * size)
        const direction = Math.random() < 0.5 ? -1 : 1
        const magnitude = parameters.standardDeviation * (3 + Math.random() * 5)
        data[index] = parameters.mean + direction * magnitude
      }
    }

    return data
  }

  // Calculate statistics from dataset
  const calculateStatistics = (data: number[]) => {
    if (!data.length) return stats

    // Sort data for easier calculations
    const sortedData = [...data].sort((a, b) => a - b)

    // Basic statistics
    const sum = sortedData.reduce((acc, val) => acc + val, 0)
    const mean = sum / sortedData.length
    const median =
      sortedData.length % 2 === 0
        ? (sortedData[sortedData.length / 2 - 1] + sortedData[sortedData.length / 2]) / 2
        : sortedData[Math.floor(sortedData.length / 2)]

    // Calculate mode
    const counts: Record<number, number> = {}
    let mode = sortedData[0]
    let maxCount = 0

    // Round to nearest integer for mode calculation
    sortedData.forEach((val) => {
      const roundedVal = Math.round(val)
      counts[roundedVal] = (counts[roundedVal] || 0) + 1
      if (counts[roundedVal] > maxCount) {
        maxCount = counts[roundedVal]
        mode = roundedVal
      }
    })

    // Variance and standard deviation
    const squaredDiffs = sortedData.map((val) => Math.pow(val - mean, 2))
    const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / sortedData.length
    const standardDeviation = Math.sqrt(variance)

    // Quartiles
    const q1Index = Math.floor(sortedData.length * 0.25)
    const q3Index = Math.floor(sortedData.length * 0.75)
    const q1 = sortedData[q1Index]
    const q3 = sortedData[q3Index]
    const iqr = q3 - q1

    // Min and max
    const min = sortedData[0]
    const max = sortedData[sortedData.length - 1]

    // Skewness (Pearson's moment coefficient of skewness)
    let skewness = 0
    if (sortedData.length > 2) {
      const cubedDiffs = sortedData.map((val) => Math.pow(val - mean, 3))
      const sumCubedDiffs = cubedDiffs.reduce((acc, val) => acc + val, 0)
      skewness = sumCubedDiffs / (sortedData.length * Math.pow(standardDeviation, 3))
    }

    // Kurtosis
    let kurtosis = 3 // Normal distribution has kurtosis of 3
    if (sortedData.length > 3) {
      const fourthPowerDiffs = sortedData.map((val) => Math.pow(val - mean, 4))
      const sumFourthPowerDiffs = fourthPowerDiffs.reduce((acc, val) => acc + val, 0)
      kurtosis = sumFourthPowerDiffs / (sortedData.length * Math.pow(standardDeviation, 4))
    }

    return {
      mean,
      median,
      mode,
      standardDeviation,
      variance,
      min,
      max,
      q1,
      q3,
      iqr,
      skewness,
      kurtosis,
    }
  }

  // Initialize charts and data
  useEffect(() => {
    // Generate initial dataset and calculate statistics
    const data = generateDataset()
    const stats = calculateStatistics(data)
    setStats(stats)

    // Set up chart rendering using canvas contexts
    if (distributionChartRef.current) {
      renderDistributionChart(distributionChartRef.current, data, stats)
    }

    if (histogramChartRef.current) {
      renderHistogramChart(histogramChartRef.current, data)
    }

    if (boxplotChartRef.current) {
      renderBoxplotChart(boxplotChartRef.current, stats)
    }

    // Cleanup on unmount
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  // Update charts and stats when parameters change
  useEffect(() => {
    const data = generateDataset()
    const newStats = calculateStatistics(data)
    setStats(newStats)

    if (distributionChartRef.current) {
      renderDistributionChart(distributionChartRef.current, data, newStats)
    }

    if (histogramChartRef.current) {
      renderHistogramChart(histogramChartRef.current, data)
    }

    if (boxplotChartRef.current) {
      renderBoxplotChart(boxplotChartRef.current, newStats)
    }
  }, [parameters, datasetType, datasetSize])

  // Animation effect
  useEffect(() => {
    if (isPlaying) {
      let lastTime = 0
      const animate = (time: number) => {
        if (!lastTime) lastTime = time
        const delta = time - lastTime

        if (delta > 1000 / speed) {
          lastTime = time

          // Gradually change parameters during animation
          setParameters((prev) => {
            const newMean = prev.mean + (Math.random() - 0.5) * 2
            const newStdDev = Math.max(1, prev.standardDeviation + (Math.random() - 0.5))

            return {
              ...prev,
              mean: newMean,
              standardDeviation: newStdDev,
            }
          })
          animationRef.current = requestAnimationFrame(animate)
        }

        animationRef.current = requestAnimationFrame(animate)
      }

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
      }
    }
  }, [speed])

  // Render distribution chart
  const renderDistributionChart = (canvas: HTMLCanvasElement, data: number[], stats: Stats) => {

  // Render distribution chart
  const renderDistributionChart = (canvas: HTMLCanvasElement, data: number[], stats: Stats) => {
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Set dimensions
    const width = canvas.width
    const height = canvas.height
    const padding = { top: 20, right: 20, bottom: 40, left: 50 }
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    // Create bins for histogram
    const binCount = 20
    const min = Math.min(...data)
    const max = Math.max(...data)
    const binWidth = (max - min) / binCount

    const bins = Array(binCount).fill(0)
    data.forEach((val) => {
      const binIndex = Math.min(Math.floor((val - min) / binWidth), binCount - 1)
      if (binIndex >= 0) bins[binIndex]++
    })

    // Find max bin height for scaling
    const maxBinHeight = Math.max(...bins)

    // Draw axes
    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(padding.left + chartWidth, height - padding.bottom)
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(padding.left, padding.top)
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw x-axis labels
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    ctx.fillStyle = "#666"
    ctx.font = "10px sans-serif"

    for (let i = 0; i <= binCount; i += 5) {
      const x = padding.left + (i / binCount) * chartWidth
      const value = min + (i / binCount) * (max - min)
      ctx.fillText(value.toFixed(0), x, height - padding.bottom + 10)
    }

    // Draw y-axis labels
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"

    for (let i = 0; i <= 5; i++) {
      const y = height - padding.bottom - (i / 5) * chartHeight
      const value = (i / 5) * maxBinHeight
      ctx.fillText(value.toFixed(0), padding.left - 10, y)
    }

    // Draw axis titles
    ctx.textAlign = "center"
    ctx.fillText("Value", padding.left + chartWidth / 2, height - 10)

    ctx.save()
    ctx.translate(15, padding.top + chartHeight / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Frequency", 0, 0)
    ctx.restore()

    // Draw bars
    ctx.fillStyle = "rgba(79, 70, 229, 0.6)"
    ctx.strokeStyle = "rgba(79, 70, 229, 1)"

    for (let i = 0; i < binCount; i++) {
      const x = padding.left + (i / binCount) * chartWidth
      const barWidth = chartWidth / binCount
      const barHeight = (bins[i] / maxBinHeight) * chartHeight

      ctx.fillRect(x, height - padding.bottom - barHeight, barWidth, barHeight)

      ctx.strokeRect(x, height - padding.bottom - barHeight, barWidth, barHeight)
    }

    // Draw mean line
    const meanX = padding.left + ((stats.mean - min) / (max - min)) * chartWidth

    ctx.beginPath()
    ctx.moveTo(meanX, height - padding.bottom)
    ctx.lineTo(meanX, padding.top)
    ctx.strokeStyle = "rgba(255, 99, 132, 0.8)"
    ctx.lineWidth = 2
    ctx.stroke()

    ctx.fillStyle = "rgba(255, 99, 132, 1)"
    ctx.textAlign = "left"
    ctx.textBaseline = "bottom"
    ctx.fillText(`Mean: ${stats.mean.toFixed(1)}`, meanX + 5, padding.top + 15)

    // Draw median line
    const medianX = padding.left + ((stats.median - min) / (max - min)) * chartWidth

    ctx.beginPath()
    ctx.moveTo(medianX, height - padding.bottom)
    ctx.lineTo(medianX, padding.top)
    ctx.strokeStyle = "rgba(54, 162, 235, 0.8)"
    ctx.lineWidth = 2
    ctx.stroke()

    ctx.fillStyle = "rgba(54, 162, 235, 1)"
    ctx.textAlign = "left"
    ctx.textBaseline = "top"
    ctx.fillText(`Median: ${stats.median.toFixed(1)}`, medianX + 5, padding.top + 30)

    // Draw mode line
    const modeX = padding.left + ((stats.mode - min) / (max - min)) * chartWidth

    ctx.beginPath()
    ctx.moveTo(modeX, height - padding.bottom)
    ctx.lineTo(modeX, padding.top)
    ctx.strokeStyle = "rgba(75, 192, 192, 0.8)"
    ctx.lineWidth = 2
    ctx.stroke()

    ctx.fillStyle = "rgba(75, 192, 192, 1)"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText(`Mode: ${stats.mode.toFixed(1)}`, modeX + 5, padding.top + 45)
  }

  // Render histogram chart (scatter plot of data points)
  const renderHistogramChart = (canvas: HTMLCanvasElement, data: number[]) => {
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Set dimensions
    const width = canvas.width
    const height = canvas.height
    const padding = { top: 20, right: 20, bottom: 40, left: 50 }
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    // Find min/max for scaling
    const min = Math.min(...data)
    const max = Math.max(...data)

    // Draw axes
    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(padding.left + chartWidth, height - padding.bottom)
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(padding.left, padding.top)
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw x-axis labels
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    ctx.fillStyle = "#666"
    ctx.font = "10px sans-serif"

    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (i / 10) * chartWidth
      const value = (i / 10) * data.length
      ctx.fillText(value.toFixed(0), x, height - padding.bottom + 10)
    }

    // Draw y-axis labels
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"

    for (let i = 0; i <= 5; i++) {
      const y = height - padding.bottom - (i / 5) * chartHeight
      const value = min + (i / 5) * (max - min)
      ctx.fillText(value.toFixed(0), padding.left - 10, y)
    }

    // Draw axis titles
    ctx.textAlign = "center"
    ctx.fillText("Index", padding.left + chartWidth / 2, height - 10)

    ctx.save()
    ctx.translate(15, padding.top + chartHeight / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Value", 0, 0)
    ctx.restore()

    // Draw data points
    ctx.fillStyle = "rgba(79, 70, 229, 0.6)"

    data.forEach((value, index) => {
      const x = padding.left + (index / data.length) * chartWidth
      const y = height - padding.bottom - ((value - min) / (max - min)) * chartHeight

      ctx.beginPath()
      ctx.arc(x, y, 3, 0, Math.PI * 2)
      ctx.fill()
    })
  }

  // Render boxplot chart
  const renderBoxplotChart = (canvas: HTMLCanvasElement, stats: Stats) => {
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Set dimensions
    const width = canvas.width
    const height = canvas.height
    const padding = { top: 40, right: 40, bottom: 40, left: 60 }
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    // Calculate scale
    const min = stats.min
    const max = stats.max
    const range = max - min

    // Function to convert value to y position
    const valueToY = (value: number) => {
      return height - padding.bottom - ((value - min) / range) * chartHeight
    }

    // Draw axes
    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(padding.left, padding.top)
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw y-axis labels
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    ctx.fillStyle = "#666"
    ctx.font = "10px sans-serif"

    for (let i = 0; i <= 5; i++) {
      const value = min + (i / 5) * range
      const y = valueToY(value)
      ctx.fillText(value.toFixed(0), padding.left - 10, y)
    }

    // Draw axis title
    ctx.save()
    ctx.translate(15, padding.top + chartHeight / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Value", 0, 0)
    ctx.restore()

    // Draw boxplot
    const boxLeft = padding.left + chartWidth * 0.3
    const boxRight = padding.left + chartWidth * 0.7
    const boxWidth = boxRight - boxLeft

    // Draw min-max line (whiskers)
    ctx.beginPath()
    ctx.moveTo(padding.left + chartWidth / 2, valueToY(stats.min))
    ctx.lineTo(padding.left + chartWidth / 2, valueToY(stats.max))
    ctx.strokeStyle = "black"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw whisker end lines
    ctx.beginPath()
    ctx.moveTo(boxLeft, valueToY(stats.min))
    ctx.lineTo(boxRight, valueToY(stats.min))
    ctx.moveTo(boxLeft, valueToY(stats.max))
    ctx.lineTo(boxRight, valueToY(stats.max))
    ctx.stroke()

    // Draw box
    ctx.beginPath()
    ctx.rect(boxLeft, valueToY(stats.q3), boxWidth, valueToY(stats.q1) - valueToY(stats.q3))
    ctx.strokeStyle = "black"
    ctx.lineWidth = 1
    ctx.stroke()
    ctx.fillStyle = "rgba(79, 70, 229, 0.3)"
    ctx.fill()

    // Draw median line
    ctx.beginPath()
    ctx.moveTo(boxLeft, valueToY(stats.median))
    ctx.lineTo(boxRight, valueToY(stats.median))
    ctx.strokeStyle = "red"
    ctx.lineWidth = 2
    ctx.stroke()

    // Add labels
    ctx.fillStyle = "black"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"

    ctx.fillText(`Max: ${stats.max.toFixed(1)}`, boxRight + 10, valueToY(stats.max))
    ctx.fillText(`Q3: ${stats.q3.toFixed(1)}`, boxRight + 10, valueToY(stats.q3))
    ctx.fillText(`Median: ${stats.median.toFixed(1)}`, boxRight + 10, valueToY(stats.median))
    ctx.fillText(`Q1: ${stats.q1.toFixed(1)}`, boxRight + 10, valueToY(stats.q1))
    ctx.fillText(`Min: ${stats.min.toFixed(1)}`, boxRight + 10, valueToY(stats.min))
  }

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleReset = () => {
    setIsPlaying(false)
    setParameters({
      mean: 50,
      standardDeviation: 15,
      skewness: 0,
      outliers: 0,
    })
  }

  const handleParameterChange = (param: string, value: number) => {
    setParameters((prev) => ({
      ...prev,
      [param]: value,
    }))
  }

  const handleDownload = (chartRef: React.RefObject<HTMLCanvasElement>, filename: string) => {
    if (!chartRef.current) return

    const link = document.createElement("a")
    link.download = filename
    link.href = chartRef.current.toDataURL("image/png")
    link.click()
  }

  return (
    <div className="space-y-6 py-4">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Dataset Insights and Statistics</h2>
          <p className="text-muted-foreground">Explore how statistical measures change with different distributions</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="bg-primary/10">
            {datasetType.charAt(0).toUpperCase() + datasetType.slice(1)} Distribution
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card className="p-6">
            <Tabs defaultValue="distribution" className="mb-4">
              <TabsList className="mb-4">
                <TabsTrigger value="distribution">Distribution</TabsTrigger>
                <TabsTrigger value="datapoints">Data Points</TabsTrigger>
                <TabsTrigger value="boxplot">Box Plot</TabsTrigger>
              </TabsList>

              <TabsContent value="distribution" className="mt-0">
                <div className="relative">
                  <div className="aspect-[16/9] bg-muted rounded-md flex items-center justify-center mb-4 overflow-hidden">
                    <canvas ref={distributionChartRef} className="w-full h-full" width={800} height={450} />
                  </div>

                  <div className="absolute top-2 right-2">
                    <Button
                      variant="secondary"
                      size="icon"
                      onClick={() => handleDownload(distributionChartRef, "distribution.png")}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="datapoints" className="mt-0">
                <div className="relative">
                  <div className="aspect-[16/9] bg-muted rounded-md flex items-center justify-center mb-4 overflow-hidden">
                    <canvas ref={histogramChartRef} className="w-full h-full" width={800} height={450} />
                  </div>

                  <div className="absolute top-2 right-2">
                    <Button
                      variant="secondary"
                      size="icon"
                      onClick={() => handleDownload(histogramChartRef, "datapoints.png")}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="boxplot" className="mt-0">
                <div className="relative">
                  <div className="aspect-[16/9] bg-muted rounded-md flex items-center justify-center mb-4 overflow-hidden">
                    <canvas ref={boxplotChartRef} className="w-full h-full" width={800} height={450} />
                  </div>

                  <div className="absolute top-2 right-2">
                    <Button
                      variant="secondary"
                      size="icon"
                      onClick={() => handleDownload(boxplotChartRef, "boxplot.png")}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </TabsContent>
            </Tabs>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Button variant="outline" size="icon" onClick={handlePlayPause}>
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
                <Button variant="outline" size="icon" onClick={handleReset}>
                  <RefreshCw className="h-4 w-4" />
                </Button>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Speed:</span>
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
          </Card>

          <Card className="p-6">
            <h3 className="font-medium mb-4">Statistical Measures</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-muted p-3 rounded-md">
                <div className="text-sm text-muted-foreground">Mean</div>
                <div className="text-2xl font-semibold">{stats.mean.toFixed(2)}</div>
              </div>
              <div className="bg-muted p-3 rounded-md">
                <div className="text-sm text-muted-foreground">Median</div>
                <div className="text-2xl font-semibold">{stats.median.toFixed(2)}</div>
              </div>
              <div className="bg-muted p-3 rounded-md">
                <div className="text-sm text-muted-foreground">Mode</div>
                <div className="text-2xl font-semibold">{stats.mode.toFixed(2)}</div>
              </div>
              <div className="bg-muted p-3 rounded-md">
                <div className="text-sm text-muted-foreground">Standard Deviation</div>
                <div className="text-2xl font-semibold">{stats.standardDeviation.toFixed(2)}</div>
              </div>
              <div className="bg-muted p-3 rounded-md">
                <div className="text-sm text-muted-foreground">Variance</div>
                <div className="text-2xl font-semibold">{stats.variance.toFixed(2)}</div>
              </div>
              <div className="bg-muted p-3 rounded-md">
                <div className="text-sm text-muted-foreground">Range</div>
                <div className="text-2xl font-semibold">{(stats.max - stats.min).toFixed(2)}</div>
              </div>
            </div>

            {showAdvancedStats && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-sm text-muted-foreground">Min</div>
                  <div className="text-2xl font-semibold">{stats.min.toFixed(2)}</div>
                </div>
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-sm text-muted-foreground">Max</div>
                  <div className="text-2xl font-semibold">{stats.max.toFixed(2)}</div>
                </div>
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-sm text-muted-foreground">Q1 (25th Percentile)</div>
                  <div className="text-2xl font-semibold">{stats.q1.toFixed(2)}</div>
                </div>
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-sm text-muted-foreground">Q3 (75th Percentile)</div>
                  <div className="text-2xl font-semibold">{stats.q3.toFixed(2)}</div>
                </div>
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-sm text-muted-foreground">IQR</div>
                  <div className="text-2xl font-semibold">{stats.iqr.toFixed(2)}</div>
                </div>
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-sm text-muted-foreground">Skewness</div>
                  <div className="text-2xl font-semibold">{stats.skewness.toFixed(2)}</div>
                </div>
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-sm text-muted-foreground">Kurtosis</div>
                  <div className="text-2xl font-semibold">{stats.kurtosis.toFixed(2)}</div>
                </div>
              </div>
            )}

            <Button variant="ghost" size="sm" className="mt-4" onClick={() => setShowAdvancedStats(!showAdvancedStats)}>
              {showAdvancedStats ? (
                <>
                  <ChevronUp className="h-4 w-4 mr-1" />
                  Hide Advanced Statistics
                </>
              ) : (
                <>
                  <ChevronDown className="h-4 w-4 mr-1" />
                  Show Advanced Statistics
                </>
              )}
            </Button>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="p-6">
            <h3 className="font-medium mb-4">Controls</h3>

            <div className="space-y-6">
              <div>
                <label className="text-sm font-medium mb-2 block">Distribution Type</label>
                <Select value={datasetType} onValueChange={setDatasetType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select distribution type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="normal">Normal Distribution</SelectItem>
                    <SelectItem value="uniform">Uniform Distribution</SelectItem>
                    <SelectItem value="bimodal">Bimodal Distribution</SelectItem>
                    <SelectItem value="exponential">Exponential Distribution</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Dataset Size</label>
                <Select value={datasetSize.toString()} onValueChange={(value) => setDatasetSize(Number(value))}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select dataset size" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="50">50 points</SelectItem>
                    <SelectItem value="100">100 points</SelectItem>
                    <SelectItem value="200">200 points</SelectItem>
                    <SelectItem value="500">500 points</SelectItem>
                    <SelectItem value="1000">1000 points</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <div className="flex justify-between">
                  <label className="text-sm font-medium mb-2 block">Mean</label>
                  <span className="text-sm text-muted-foreground">{parameters.mean.toFixed(1)}</span>
                </div>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[parameters.mean]}
                    min={0}
                    max={100}
                    step={1}
                    className="flex-1"
                    onValueChange={(value) => handleParameterChange("mean", value[0])}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between">
                  <label className="text-sm font-medium mb-2 block">Standard Deviation</label>
                  <span className="text-sm text-muted-foreground">{parameters.standardDeviation.toFixed(1)}</span>
                </div>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[parameters.standardDeviation]}
                    min={1}
                    max={30}
                    step={0.5}
                    className="flex-1"
                    onValueChange={(value) => handleParameterChange("standardDeviation", value[0])}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between">
                  <label className="text-sm font-medium mb-2 block">Skewness</label>
                  <span className="text-sm text-muted-foreground">{parameters.skewness.toFixed(1)}</span>
                </div>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[parameters.skewness]}
                    min={-2}
                    max={2}
                    step={0.1}
                    className="flex-1"
                    onValueChange={(value) => handleParameterChange("skewness", value[0])}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between">
                  <label className="text-sm font-medium mb-2 block">Outliers (%)</label>
                  <span className="text-sm text-muted-foreground">{parameters.outliers.toFixed(0)}%</span>
                </div>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[parameters.outliers]}
                    min={0}
                    max={20}
                    step={1}
                    className="flex-1"
                    onValueChange={(value) => handleParameterChange("outliers", value[0])}
                  />
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="font-medium mb-4">Explanation</h3>
            <p className="text-muted-foreground mb-4">
              This demo visualizes how statistical measures change as you adjust distribution parameters. Experiment
              with different distributions and parameters to see their effect on mean, median, mode, and other
              statistics.
            </p>
            <div className="grid grid-cols-1 gap-4">
              <div>
                <h4 className="text-sm font-medium mb-2">Key Observations</h4>
                <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                  <li>In a normal distribution, mean = median = mode</li>
                  <li>Skewness affects the relationship between mean, median, and mode</li>
                  <li>Outliers have a stronger effect on mean than on median</li>
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-medium mb-2">Tips</h4>
                <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                  <li>Try different distribution types to see how they affect the statistics</li>
                  <li>Add outliers to see how they impact the mean vs. median</li>
                  <li>Adjust skewness to see how it shifts the distribution</li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

function renderHistogramChart(current: HTMLCanvasElement, data: number[]) {
        throw new Error("Function not implemented.")
    }
    function renderBoxplotChart(current: HTMLCanvasElement, newStats: { mean: number; median: number; mode: number; standardDeviation: number; variance: number; min: number; max: number; q1: number; q3: number; iqr: number; skewness: number; kurtosis: number }) {
        throw new Error("Function not implemented.")
    }
}
