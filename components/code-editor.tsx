"use client"

import React, { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { PlusCircle, Play, Trash2, AlertCircle, Loader2 } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { getDatasetsByAlgorithm, type Dataset } from "@/lib/datasets"
import { Chart } from "@/components/ui/chart"

interface PythonCodeEditorProps {
  algorithm: string
}

interface CodeCell {
  id: string
  code: string
  output: string
  error: string | null
  visualization: any
  isExecuting: boolean
}

export function PythonCodeEditor({ algorithm }: PythonCodeEditorProps) {
  const [cells, setCells] = useState<CodeCell[]>([
    {
      id: "1",
      code: "",
      output: "",
      error: null,
      visualization: null,
      isExecuting: false,
    },
  ])
  const [selectedDataset, setSelectedDataset] = useState<string>("")
  const [availableDatasets, setAvailableDatasets] = useState<Dataset[]>([])

  // Fetch datasets for this algorithm
  React.useEffect(() => {
    const datasets = getDatasetsByAlgorithm(algorithm)
    setAvailableDatasets(datasets)
    if (datasets.length > 0) {
      setSelectedDataset(datasets[0].id)
      // Pre-populate first cell with dataset preview code
      setCells([
        {
          id: "1",
          code: datasets[0].previewCode,
          output: "",
          error: null,
          visualization: null,
          isExecuting: false,
        },
      ])
    }
  }, [algorithm])

  const addCell = () => {
    const newId = Date.now().toString()
    setCells([
      ...cells,
      {
        id: newId,
        code: "",
        output: "",
        error: null,
        visualization: null,
        isExecuting: false,
      },
    ])
  }

  const removeCell = (id: string) => {
    if (cells.length > 1) {
      setCells(cells.filter((cell) => cell.id !== id))
    }
  }

  const updateCellCode = (id: string, code: string) => {
    setCells(cells.map((cell) => (cell.id === id ? { ...cell, code } : cell)))
  }

  const executeCell = async (id: string) => {
    // Find the cell
    const cellIndex = cells.findIndex((cell) => cell.id === id)
    if (cellIndex === -1) return

    // Mark cell as executing
    setCells(cells.map((cell) => (cell.id === id ? { ...cell, isExecuting: true, output: "", error: null } : cell)))

    try {
      const response = await fetch("/api/python", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          code: cells[cellIndex].code,
          dataset: selectedDataset,
          algorithm,
        }),
      })

      const result = await response.json()

      // Update cell with results
      setCells(
        cells.map((cell) =>
          cell.id === id
            ? {
                ...cell,
                output: result.output || "",
                error: result.error,
                visualization: result.visualization,
                isExecuting: false,
              }
            : cell,
        ),
      )
    } catch (error) {
      console.error("Error executing code:", error)
      setCells(
        cells.map((cell) =>
          cell.id === id
            ? {
                ...cell,
                error: "Failed to execute code. Network error or server issue.",
                isExecuting: false,
              }
            : cell,
        ),
      )
    }
  }

  const handleDatasetChange = (datasetId: string) => {
    setSelectedDataset(datasetId)

    // Update first cell with dataset preview code
    const dataset = availableDatasets.find((d) => d.id === datasetId)
    if (dataset && cells.length > 0) {
      setCells(cells.map((cell, index) => (index === 0 ? { ...cell, code: dataset.previewCode } : cell)))
    }
  }

  // Function to render HTML table safely
  const renderHtmlTable = (htmlString: string) => {
    return <div className="overflow-auto" dangerouslySetInnerHTML={{ __html: htmlString }} />
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Select value={selectedDataset} onValueChange={handleDatasetChange}>
            <SelectTrigger className="w-[250px]">
              <SelectValue placeholder="Select a dataset" />
            </SelectTrigger>
            <SelectContent>
              {availableDatasets.map((dataset) => (
                <SelectItem key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {selectedDataset && (
            <p className="text-sm text-muted-foreground">
              {availableDatasets.find((d) => d.id === selectedDataset)?.description}
            </p>
          )}
        </div>

        <Button onClick={addCell} variant="outline" size="sm">
          <PlusCircle className="h-4 w-4 mr-2" />
          Add Cell
        </Button>
      </div>

      <div className="space-y-6">
        {cells.map((cell, index) => (
          <Card key={cell.id} className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm text-muted-foreground">Cell {index + 1}</div>
              <div className="flex items-center gap-2">
                <Button onClick={() => executeCell(cell.id)} size="sm" variant="outline" disabled={cell.isExecuting}>
                  {cell.isExecuting ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Run
                </Button>

                {cells.length > 1 && (
                  <Button onClick={() => removeCell(cell.id)} size="sm" variant="outline" disabled={cell.isExecuting}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>

            <Textarea
              value={cell.code}
              onChange={(e) => updateCellCode(cell.id, e.target.value)}
              className="font-mono text-sm min-h-[150px] mb-4"
              placeholder="# Write your Python code here..."
            />

            {(cell.output || cell.error || cell.visualization) && (
              <div className="mt-4 border-t pt-4">
                <div className="text-sm font-medium mb-2">Output:</div>

                {cell.error ? (
                  <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-900 p-3 rounded-md">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="h-5 w-5 text-red-500 mt-0.5" />
                      <pre className="text-sm text-red-700 dark:text-red-300 whitespace-pre-wrap font-mono">
                        {cell.error}
                      </pre>
                    </div>
                  </div>
                ) : (
                  <>
                    {cell.output && (
                      <pre className="bg-muted p-3 rounded-md text-sm font-mono whitespace-pre-wrap mb-4">
                        {cell.output}
                      </pre>
                    )}

                    {cell.visualization && (
                      <div className="mt-4">
                        {cell.visualization.type === "html_table" ? (
                          renderHtmlTable(cell.visualization.data)
                        ) : cell.visualization.type === "plot" ? (
                          <div className="flex justify-center">
                            <img
                              src={cell.visualization.data || "/placeholder.svg"}
                              alt="Plot visualization"
                              className="max-w-full h-auto border rounded-md"
                            />
                          </div>
                        ) : cell.visualization.type === "table" ? (
                          <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                              <thead className="bg-gray-50 dark:bg-gray-800">
                                <tr>
                                  {cell.visualization.data.columns.map((column: string, i: number) => (
                                    <th
                                      key={i}
                                      scope="col"
                                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                                    >
                                      {column}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                                {cell.visualization.data.rows.map((row: any[], rowIndex: number) => (
                                  <tr key={rowIndex}>
                                    {row.map((cell: any, cellIndex: number) => (
                                      <td
                                        key={cellIndex}
                                        className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300"
                                      >
                                        {cell}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        ) : (
                          <Chart type={cell.visualization.type} data={cell.visualization.data} />
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </Card>
        ))}
      </div>
    </div>
  )
}
