"use client"

import type { Algorithm } from "@/lib/types"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useState, useEffect } from "react"
import {
  CheckCircle,
  ArrowRight,
  Play,
  Code,
  XCircle,
  Plus,
  Trash2,
  ArrowUp,
  ArrowDown,
  Save,
  Download,
  FileCode,
} from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { NotebookCell } from "@/components/notebook-cell"
import { getPracticeSteps } from "@/lib/practice-steps"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { v4 as uuidv4 } from "uuid"

interface PracticeTabProps {
  algorithm: Algorithm
}

export interface Cell {
  id: string
  code: string
  result: null | {
    success: boolean
    message: string
    output?: string
    error?: string
    table_html?: string
    plot?: string
  }
  isExecuting: boolean
  executionCount: number | null
}

export function PracticeTab({ algorithm }: PracticeTabProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [cells, setCells] = useState<Cell[]>([])
  const [activeTab, setActiveTab] = useState("notebook")
  const [isSolutionCorrect, setIsSolutionCorrect] = useState<boolean | null>(null)
  const [executionCounter, setExecutionCounter] = useState(1)
  const [selectedCellIndex, setSelectedCellIndex] = useState<number | null>(null)

  const steps = getPracticeSteps(algorithm.slug)

  useEffect(() => {
    // Initialize with starter code when step changes
    if (steps && steps[currentStep]) {
      // Create a single cell with the starter code
      setCells([
        {
          id: uuidv4(),
          code: steps[currentStep].starterCode || "",
          result: null,
          isExecuting: false,
          executionCount: null,
        },
      ])
      setIsSolutionCorrect(null)
      setExecutionCounter(1)
      setSelectedCellIndex(0)
    }
  }, [currentStep, steps])

  const checkSolution = (results: any[]) => {
    if (!results || results.length === 0 || results.some((r) => r.error)) {
      setIsSolutionCorrect(false)
      return
    }

    // Check if any output contains expected elements
    const hasDataFrame = results.some((r) => (r.output && r.output.includes("DataFrame")) || r.table_html)

    const hasHead = results.some((r) => r.output && r.output.toLowerCase().includes("first 5 rows"))

    const hasPlot = results.some((r) => r.plot)

    // Adjust this logic based on the current step requirements
    if (currentStep === 0) {
      setIsSolutionCorrect(hasDataFrame && hasHead)
    } else if (currentStep === 3) {
      setIsSolutionCorrect(hasPlot)
    } else {
      setIsSolutionCorrect(results.every((r) => r.success))
    }
  }

  const executeCell = async (cellIndex: number) => {
    const cell = cells[cellIndex]
    if (!cell || cell.isExecuting) return

    // Update the cell to show it's executing
    setCells((prev) => prev.map((c, i) => (i === cellIndex ? { ...c, isExecuting: true, result: null } : c)))

    try {
      const response = await fetch("/api/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          cells: [{ code: cell.code }],
          algorithm: algorithm.slug,
        }),
      })

      const data = await response.json()

      if (data && data.length > 0) {
        const result = data[0]

        // Update the cell with the result and execution count
        setCells((prev) =>
          prev.map((c, i) =>
            i === cellIndex
              ? {
                  ...c,
                  isExecuting: false,
                  result: {
                    success: !result.error,
                    message: result.error ? `Error: ${result.error}` : "Code executed successfully!",
                    output: result.output,
                    error: result.error,
                    table_html: result.table_html,
                    plot: result.plot,
                  },
                  executionCount: executionCounter,
                }
              : c,
          ),
        )

        // Increment the execution counter for the next cell
        setExecutionCounter((prev) => prev + 1)
      }
    } catch (error) {
      setCells((prev) =>
        prev.map((c, i) =>
          i === cellIndex
            ? {
                ...c,
                isExecuting: false,
                result: {
                  success: false,
                  message: `Failed to execute code: ${error instanceof Error ? error.message : String(error)}`,
                },
                executionCount: executionCounter,
              }
            : c,
        ),
      )

      setExecutionCounter((prev) => prev + 1)
    }
  }

  const executeAllCells = async () => {
    // Execute cells in sequence
    for (let i = 0; i < cells.length; i++) {
      await executeCell(i)
    }

    // Check solution after all cells have executed
    const results = cells.map((cell) => cell.result).filter(Boolean) as any[]
    checkSolution(results)
  }

  const addCell = (index: number) => {
    setCells((prev) => [
      ...prev.slice(0, index + 1),
      {
        id: uuidv4(),
        code: "",
        result: null,
        isExecuting: false,
        executionCount: null,
      },
      ...prev.slice(index + 1),
    ])

    // Select the new cell
    setSelectedCellIndex(index + 1)
  }

  const deleteCell = (index: number) => {
    // Don't delete if it's the only cell
    if (cells.length <= 1) return

    setCells((prev) => [...prev.slice(0, index), ...prev.slice(index + 1)])

    // Update selected cell
    if (selectedCellIndex === index) {
      setSelectedCellIndex(Math.min(index, cells.length - 2))
    } else if (selectedCellIndex && selectedCellIndex > index) {
      setSelectedCellIndex(selectedCellIndex - 1)
    }
  }

  const moveCell = (index: number, direction: "up" | "down") => {
    if (direction === "up" && index === 0) return
    if (direction === "down" && index === cells.length - 1) return

    const newIndex = direction === "up" ? index - 1 : index + 1

    setCells((prev) => {
      const newCells = [...prev]
      const [movedCell] = newCells.splice(index, 1)
      newCells.splice(newIndex, 0, movedCell)
      return newCells
    })

    // Update selected cell
    if (selectedCellIndex === index) {
      setSelectedCellIndex(newIndex)
    } else if (selectedCellIndex === newIndex) {
      setSelectedCellIndex(index)
    }
  }

  const updateCellCode = (index: number, code: string) => {
    setCells((prev) => prev.map((cell, i) => (i === index ? { ...cell, code } : cell)))
  }

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const downloadNotebook = () => {
    const notebookData = {
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3",
        },
        language_info: {
          codemirror_mode: {
            name: "ipython",
            version: 3,
          },
          file_extension: ".py",
          mimetype: "text/x-python",
          name: "python",
          nbconvert_exporter: "python",
          pygments_lexer: "ipython3",
          version: "3.8.10",
        },
      },
      nbformat: 4,
      nbformat_minor: 5,
      cells: cells.map((cell) => ({
        cell_type: "code",
        execution_count: cell.executionCount,
        metadata: {},
        source: cell.code.split("\n"),
        outputs: cell.result
          ? [
              {
                name: "stdout",
                output_type: "stream",
                text: cell.result.output ? cell.result.output.split("\n") : [],
              },
            ]
          : [],
      })),
    }

    const blob = new Blob([JSON.stringify(notebookData, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${algorithm.slug}-practice-step-${currentStep + 1}.ipynb`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const resetStep = () => {
    if (steps && steps[currentStep]) {
      setCells([
        {
          id: uuidv4(),
          code: steps[currentStep].starterCode || "",
          result: null,
          isExecuting: false,
          executionCount: null,
        },
      ])
      setIsSolutionCorrect(null)
      setExecutionCounter(1)
      setSelectedCellIndex(0)
    }
  }

  if (!steps || steps.length === 0) {
    return (
      <div className="py-8 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-muted mb-4">
          <Code className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-xl font-medium mb-2">Practice Coming Soon</h3>
        <p className="text-muted-foreground max-w-md mx-auto">
          We're currently developing practice exercises for {algorithm.title}. Check back soon for interactive coding
          challenges!
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6 py-4">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <h2 className="text-2xl font-bold">{algorithm.title} Practice</h2>
          {isSolutionCorrect !== null &&
            (isSolutionCorrect ? (
              <Badge variant="success" className="flex items-center gap-1">
                <CheckCircle className="h-3.5 w-3.5" />
                <span>Correct Solution</span>
              </Badge>
            ) : (
              <Badge variant="destructive" className="flex items-center gap-1">
                <XCircle className="h-3.5 w-3.5" />
                <span>Try Again</span>
              </Badge>
            ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Progress:</span>
          <Progress value={(currentStep / (steps.length - 1)) * 100} className="w-40" />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1">
          <Card className="p-4 h-full">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-medium">{steps[currentStep].title}</h3>
              <Badge variant="outline">
                Step {currentStep + 1}/{steps.length}
              </Badge>
            </div>

            <Tabs defaultValue="instructions" className="w-full">
              <TabsList className="grid w-full grid-cols-1">
                <TabsTrigger value="instructions">Instructions</TabsTrigger>
              </TabsList>

              <TabsContent value="instructions" className="space-y-4 mt-4">
                <div className="prose dark:prose-invert prose-sm max-w-none">
                  <div dangerouslySetInnerHTML={{ __html: steps[currentStep].instruction }} />
                </div>

                <Separator />

                <div>
                  <h4 className="text-sm font-medium mb-2">Hints</h4>
                  <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                    {steps[currentStep].hints.map((hint, index) => (
                      <li key={index}>{hint}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="text-sm font-medium mb-2">Resources</h4>
                  <ul className="text-sm space-y-1">
                    {steps[currentStep].resources.map((resource, index) => (
                      <li key={index}>
                        <Button variant="link" className="p-0 h-auto text-sm" asChild>
                          <a href={resource.url} target="_blank" rel="noopener noreferrer">
                            {resource.title}
                          </a>
                        </Button>
                      </li>
                    ))}
                  </ul>
                </div>
              </TabsContent>
            </Tabs>
          </Card>
        </div>

        <div className="md:col-span-2">
          <Card className="p-0 overflow-hidden">
            <div className="bg-muted p-2 border-b flex items-center justify-between">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList>
                  <TabsTrigger value="notebook" className="text-xs">
                    <FileCode className="h-4 w-4 mr-1" />
                    Notebook
                  </TabsTrigger>
                </TabsList>
              </Tabs>

              <div className="flex items-center gap-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon" onClick={executeAllCells}>
                        <Play className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Run All Cells</TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon" onClick={resetStep}>
                        <Save className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Reset to Starter Code</TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon" onClick={downloadNotebook}>
                        <Download className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Download as Jupyter Notebook</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </div>

            <div className="p-4 notebook-container">
              {cells.map((cell, index) => (
                <div key={cell.id} className="mb-4 relative">
                  <NotebookCell
                    cell={cell}
                    isSelected={selectedCellIndex === index}
                    onSelect={() => setSelectedCellIndex(index)}
                    onChange={(code) => updateCellCode(index, code)}
                    onExecute={() => executeCell(index)} />

                  {selectedCellIndex === index && (
                    <div className="absolute -left-10 top-2 flex flex-col gap-1">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={() => moveCell(index, "up")}
                              disabled={index === 0}
                            >
                              <ArrowUp className="h-3 w-3" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent side="left">Move Up</TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={() => moveCell(index, "down")}
                              disabled={index === cells.length - 1}
                            >
                              <ArrowDown className="h-3 w-3" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent side="left">Move Down</TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={() => deleteCell(index)}
                              disabled={cells.length <= 1}
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent side="left">Delete Cell</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  )}

                  {/* Add cell button */}
                  <div className="flex justify-center">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 rounded-full opacity-50 hover:opacity-100"
                      onClick={() => addCell(index)}
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            <div className="p-4 border-t bg-muted flex items-center justify-between">
              <Button variant="outline" onClick={handlePrevious} disabled={currentStep === 0}>
                Previous
              </Button>

              <Button onClick={handleNext} disabled={currentStep === steps.length - 1}>
                Next <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

