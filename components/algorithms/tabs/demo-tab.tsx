"use client"

import type { Algorithm } from "@/lib/types"
import { Card } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useState } from "react"
import { Play, Pause, RefreshCw } from "lucide-react"
import { EnhancedEdaDemo } from "./enhanced-eda-demo"
interface DemoTabProps {
  algorithm: Algorithm
}

export function DemoTab({ algorithm }: DemoTabProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [dataset, setDataset] = useState("default")
  const [parameters, setParameters] = useState({
    param1: 50,
    param2: 30,
  })

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleReset = () => {
    setIsPlaying(false)
    // Reset visualization state
  }

  const handleParameterChange = (param: string, value: number) => {
    setParameters((prev) => ({
      ...prev,
      [param]: value,
    }))
  }

  // Render the appropriate demo based on algorithm
  const renderAlgorithmDemo = () => {
    switch (algorithm.slug) {
      case "eda":
        return <EnhancedEdaDemo />
      default:
        return (
          <div className="space-y-6">
            <Card className="p-6">
              <div className="aspect-video bg-muted rounded-md flex items-center justify-center mb-4">
                <p className="text-muted-foreground">Visualization would render here</p>
              </div>

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
              <h3 className="font-medium mb-4">Controls</h3>

              <div className="space-y-6">
                <div>
                  <label className="text-sm font-medium mb-2 block">Dataset</label>
                  <Select value={dataset} onValueChange={setDataset}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">Default Dataset</SelectItem>
                      <SelectItem value="simple">Simple Dataset</SelectItem>
                      <SelectItem value="complex">Complex Dataset</SelectItem>
                      <SelectItem value="custom">Custom Dataset</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Parameter 1</label>
                  <div className="flex items-center gap-4">
                    <Slider
                      value={[parameters.param1]}
                      min={0}
                      max={100}
                      step={1}
                      className="flex-1"
                      onValueChange={(value) => handleParameterChange("param1", value[0])}
                    />
                    <span className="text-sm w-8 text-right">{parameters.param1}</span>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium mb-2 block">Parameter 2</label>
                  <div className="flex items-center gap-4">
                    <Slider
                      value={[parameters.param2]}
                      min={0}
                      max={100}
                      step={1}
                      className="flex-1"
                      onValueChange={(value) => handleParameterChange("param2", value[0])}
                    />
                    <span className="text-sm w-8 text-right">{parameters.param2}</span>
                  </div>
                </div>

                <div className="pt-4">
                  <h4 className="text-sm font-medium mb-2">Results</h4>
                  <div className="bg-muted p-3 rounded-md">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Accuracy:</div>
                      <div className="text-right font-medium">87%</div>
                      <div>Error Rate:</div>
                      <div className="text-right font-medium">0.13</div>
                      <div>Iterations:</div>
                      <div className="text-right font-medium">24</div>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        )
    }
  }

  return (
    <div className="space-y-6 py-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">{algorithm.title} Demo</h2>
      </div>

      {renderAlgorithmDemo()}

      {algorithm.slug !== "eda" && (
        <Card className="p-6">
          <h3 className="font-medium mb-4">Explanation</h3>
          <p className="text-muted-foreground">
            This demo visualizes how {algorithm.title} works with different parameters and datasets. Adjust the controls
            to see how the algorithm behaves under different conditions.
          </p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium mb-2">Key Observations</h4>
              <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                <li>Observation 1 about the algorithm behavior</li>
                <li>Observation 2 about parameter sensitivity</li>
                <li>Observation 3 about dataset characteristics</li>
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-medium mb-2">Tips</h4>
              <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                <li>Try different datasets to see how the algorithm adapts</li>
                <li>Adjust parameters to optimize performance</li>
                <li>Watch for patterns in the visualization</li>
              </ul>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}

