"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Play, ChevronDown, ChevronRight, Loader2, AlertCircle, Trash2, Database } from "lucide-react"
import type { Cell } from "@/components/algorithms/tabs/practice-tab"
import { cn } from "@/lib/utils"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { getDatasetsByAlgorithm } from "@/lib/datasets"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"

interface NotebookCellProps {
  cell: Cell
  isSelected: boolean
  onSelect: () => void
  onChange: (code: string) => void
  onExecute: () => void
  algorithm?: string
  cellIndex?: number
  onRemove?: () => void
  showDatasetSelector?: boolean
  currentStep?: number
}

export function NotebookCell({
  cell,
  isSelected,
  onSelect,
  onChange,
  onExecute,
  algorithm = "",
  cellIndex = 0,
  onRemove,
  showDatasetSelector = false,
  currentStep = 0,
}: NotebookCellProps) {
  const [isOutputCollapsed, setIsOutputCollapsed] = useState(false)
  const [selectedDataset, setSelectedDataset] = useState<string>("")
  const [availableDatasets, setAvailableDatasets] = useState<any[]>([])
  const [textareaHeight, setTextareaHeight] = useState("150px")
  const editorRef = useRef<any>(null)
  const editorContainerRef = useRef<HTMLDivElement>(null)

  // Fetch datasets for this algorithm
  useEffect(() => {
    if (algorithm) {
      const datasets = getDatasetsByAlgorithm(algorithm)
      setAvailableDatasets(datasets)
      if (datasets.length > 0 && !selectedDataset) {
        setSelectedDataset(datasets[0].id)
        // Only pre-populate first cell with dataset preview code if showing dataset selector
        if (cellIndex === 0 && !cell.code && showDatasetSelector) {
          onChange(datasets[0].previewCode)
        }
      }
    }
  }, [algorithm, cellIndex, onChange, cell.code, selectedDataset, showDatasetSelector])

  // Adjust textarea height based on content
  useEffect(() => {
    const lineCount = (cell.code.match(/\n/g) || []).length + 1
    const newHeight = Math.max(150, Math.min(500, lineCount * 20))
    setTextareaHeight(`${newHeight}px`)
  }, [cell.code])

  const handleDatasetChange = (datasetId: string) => {
    setSelectedDataset(datasetId)

    // Only update code if this is the first cell and we're showing the dataset selector
    if (cellIndex === 0 && showDatasetSelector) {
      const dataset = availableDatasets.find((d) => d.id === datasetId)
      if (dataset) {
        onChange(dataset.previewCode)
      }
    }
  }

  // Function to render HTML table safely
  const renderHtmlTable = (htmlString: string) => {
    return <div className="overflow-auto" dangerouslySetInnerHTML={{ __html: enhanceTableHtml(htmlString) }} />
  }

  // Helper function to enhance table HTML with responsive classes
  function enhanceTableHtml(tableHtml: string): string {
    // Add responsive table classes to the table element
    return tableHtml
      .replace(/<table/g, '<table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700 table-auto"')
      .replace(/<thead/g, '<thead class="bg-gray-50 dark:bg-gray-800"')
      .replace(
        /<th/g,
        '<th class="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"',
      )
      .replace(/<tbody/g, '<tbody class="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800"')
      .replace(/<tr>/g, '<tr class="hover:bg-gray-50 dark:hover:bg-gray-800">')
      .replace(/<td/g, '<td class="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400"')
  }

  return (
    <Card
      className={cn(
        "overflow-hidden transition-all",
        isSelected ? "ring-2 ring-primary" : "hover:ring-1 hover:ring-primary/50",
      )}
      onClick={onSelect}
    >
      {/* Cell header */}
      <div className="bg-muted/50 p-3 flex items-center gap-2 border-b">
        <div className="flex items-center justify-center w-6 h-6 bg-primary/10 rounded text-xs font-mono">
          {cell.executionCount !== null ? `[${cell.executionCount}]` : `[${cellIndex + 1}]`}
        </div>

        {/* Dataset selector - only show if enabled and for first cell */}
        {showDatasetSelector && cellIndex === 0 && algorithm && availableDatasets.length > 0 && (
          <div className="ml-2" onClick={(e) => e.stopPropagation()}>
            <Select value={selectedDataset} onValueChange={handleDatasetChange}>
              <SelectTrigger className="h-8 w-[200px] text-xs">
                <Database className="h-3.5 w-3.5 mr-1.5" />
                <SelectValue placeholder="Select dataset" />
              </SelectTrigger>
              <SelectContent>
                {availableDatasets.map((dataset) => (
                  <SelectItem key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {/* Dataset description */}
        {showDatasetSelector && selectedDataset && (
          <p className="text-xs text-muted-foreground hidden md:block">
            {availableDatasets.find((d) => d.id === selectedDataset)?.description}
          </p>
        )}

        <div className="flex-1"></div>

        {/* Cell actions */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="h-8"
            onClick={(e) => {
              e.stopPropagation()
              onExecute()
            }}
            disabled={cell.isExecuting}
          >
            {cell.isExecuting ? (
              <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5 mr-1.5" />
            )}
            Run
          </Button>

          {onRemove && (
            <Button
              variant="outline"
              size="sm"
              className="h-8"
              onClick={(e) => {
                e.stopPropagation()
                onRemove()
              }}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>

      {/* Code editor */}
      <div className="p-4" onClick={(e) => e.stopPropagation()} ref={editorContainerRef}>
        <Textarea
          value={cell.code}
          onChange={(e) => onChange(e.target.value)}
          className="font-mono text-sm w-full resize-none"
          placeholder="# Write your Python code here..."
          style={{ height: textareaHeight }}
        />
      </div>

      {/* Cell output area */}
      {cell.result && (
        <div className="border-t">
          <div
            className="bg-muted/30 p-3 flex items-center gap-2 cursor-pointer"
            onClick={(e) => {
              e.stopPropagation()
              setIsOutputCollapsed(!isOutputCollapsed)
            }}
          >
            {isOutputCollapsed ? (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
            <span className="text-sm font-medium">
              Output {cell.executionCount !== null ? `[${cell.executionCount}]` : ""}
            </span>
          </div>

          {!isOutputCollapsed && (
            <div className="p-4">
              {/* Error output */}
              {cell.result.error && (
                <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-900 p-3 rounded-md mb-4">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
                    <pre className="text-sm text-red-700 dark:text-red-300 whitespace-pre-wrap font-mono overflow-auto max-h-[300px]">
                      {cell.result.error}
                    </pre>
                  </div>
                </div>
              )}

              {/* Text output */}
              {cell.result.output && cell.result.output.trim() && (
                <pre className="bg-muted p-3 rounded-md text-sm font-mono whitespace-pre-wrap mb-4 overflow-auto max-h-[300px]">
                  {cell.result.output}
                </pre>
              )}

              {/* Table output */}
              {cell.result.table_html && (
                <div className="mb-4 relative">
                  <div className="overflow-x-auto max-w-full">
                    <div className="inline-block min-w-full align-middle">
                      <div
                        className="overflow-hidden border rounded-lg"
                        dangerouslySetInnerHTML={{
                          __html: enhanceTableHtml(cell.result.table_html),
                        }}
                      />
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1 text-right italic">
                    Scroll horizontally to view more data
                  </div>
                </div>
              )}

              {/* Plot output */}
              {cell.result.plot && (
                <div className="flex justify-center mb-4">
                  <img
                    src={`data:image/png;base64,${cell.result.plot}`}
                    alt="Plot"
                    className="max-w-full h-auto rounded border"
                  />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

