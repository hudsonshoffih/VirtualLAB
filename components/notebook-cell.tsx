"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { CodeEditor } from "@/components/code-editor"
import { Play, ChevronDown, ChevronRight } from "lucide-react"
import type { Cell } from "@/components/algorithms/tabs/practice-tab"
import { cn } from "@/lib/utils"

interface NotebookCellProps {
  cell: Cell
  isSelected: boolean
  onSelect: () => void
  onChange: (code: string) => void
  onExecute: () => void
}

export function NotebookCell({ cell, isSelected, onSelect, onChange, onExecute }: NotebookCellProps) {
  const [isOutputCollapsed, setIsOutputCollapsed] = useState(false)

  return (
    <div
      className={cn(
        "border rounded-md overflow-hidden transition-all",
        isSelected ? "ring-2 ring-primary" : "hover:border-primary/50",
      )}
      onClick={onSelect}
    >
      {/* Cell input area */}
      <div className="bg-muted/50 p-2 flex items-center gap-2 border-b">
        <div className="flex items-center justify-center w-6 h-6 bg-primary/10 rounded text-xs font-mono">
          {cell.executionCount !== null ? `[${cell.executionCount}]` : "[ ]"}
        </div>
        <span className="text-xs text-muted-foreground">In:</span>
        <div className="flex-1"></div>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={(e) => {
            e.stopPropagation()
            onExecute()
          }}
          disabled={cell.isExecuting}
        >
          <Play className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* Code editor */}
      <div className="p-0" onClick={(e) => e.stopPropagation()}>
        <CodeEditor value={cell.code} onChange={onChange} language="python" height="100px" />
      </div>

      {/* Cell output area */}
      {cell.result && (
        <div className="border-t">
          <div
            className="bg-muted/30 p-2 flex items-center gap-2 cursor-pointer"
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
            <span className="text-xs text-muted-foreground">Out [{cell.executionCount}]:</span>
          </div>

          {!isOutputCollapsed && (
            <div className="p-4">
              {/* Text output */}
              {cell.result.output && cell.result.output.trim() && (
                <pre className="text-sm font-mono whitespace-pre-wrap mb-4 overflow-auto max-h-[300px]">
                  {cell.result.output}
                </pre>
              )}

              {/* Error output */}
              {cell.result.error && (
                <pre className="text-sm font-mono whitespace-pre-wrap mb-4 text-red-500 bg-red-50 dark:bg-red-950 p-2 rounded overflow-auto max-h-[300px]">
                  {cell.result.error}
                </pre>
              )}

              {/* Table output */}
              {cell.result.table_html && (
                <div
                  className="mb-4 overflow-auto max-h-[300px]"
                  dangerouslySetInnerHTML={{ __html: cell.result.table_html }}
                />
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
    </div>
  )
}

