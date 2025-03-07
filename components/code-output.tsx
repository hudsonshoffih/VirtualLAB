"use client"

import { useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Maximize2, Minimize2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface CodeOutputProps {
  result: {
    success?: boolean
    message?: string
    output?: string
    error?: string
    table_html?: string
    plot?: string
  } | null
  isExecuting: boolean
  cellId: string
}

export function CodeOutput({ result, isExecuting, cellId }: CodeOutputProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState<string>("output")

  if (isExecuting) {
    return (
      <div className="space-y-2">
        <p className="text-sm text-muted-foreground">Executing code...</p>
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-5/6" />
      </div>
    )
  }

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-6 text-center">
        <p className="text-sm text-muted-foreground">Run your code to see the output here</p>
      </div>
    )
  }

  const hasOutput = result.output && result.output.trim().length > 0
  const hasTable = result.table_html && result.table_html.trim().length > 0
  const hasPlot = result.plot && result.plot.trim().length > 0
  const hasError = result.error && result.error.trim().length > 0

  if (!hasOutput && !hasTable && !hasPlot && !hasError) {
    return (
      <div className="py-2">
        <p className="text-sm text-muted-foreground">
          {result.success ? "Code executed successfully with no output." : "No output available."}
        </p>
      </div>
    )
  }

  // Determine which tabs to show
  const tabs = []
  if (hasOutput) tabs.push({ id: "output", label: "Output" })
  if (hasTable) tabs.push({ id: "table", label: "Table" })
  if (hasPlot) tabs.push({ id: "plot", label: "Plot" })

  // Set max height based on expansion state
  const maxHeight = isExpanded ? "none" : "300px"

  return (
    <div className="relative">
      <div className="absolute right-2 top-2 z-10">
        <Button variant="ghost" size="icon" onClick={() => setIsExpanded(!isExpanded)} className="h-6 w-6">
          {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
        </Button>
      </div>

      <div className={cn("transition-all duration-200", isExpanded ? "min-h-[300px]" : "max-h-[300px]")}>
        {tabs.length > 0 ? (
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full" style={{ gridTemplateColumns: `repeat(${tabs.length}, 1fr)` }}>
              {tabs.map((tab) => (
                <TabsTrigger key={tab.id} value={tab.id}>
                  {tab.label}
                </TabsTrigger>
              ))}
            </TabsList>

            {hasOutput && (
              <TabsContent value="output" className="mt-2">
                <div
                  className="bg-muted p-4 rounded-md text-sm font-mono overflow-auto whitespace-pre-wrap"
                  style={{ maxHeight }}
                >
                  {result.output}
                </div>
              </TabsContent>
            )}

            {hasTable && (
              <TabsContent value="table" className="mt-2">
                <div
                  className="bg-muted p-4 rounded-md overflow-auto"
                  style={{ maxHeight }}
                  dangerouslySetInnerHTML={{
                    __html: result.table_html || "",
                  }}
                />
              </TabsContent>
            )}

            {hasPlot && (
              <TabsContent value="plot" className="mt-2">
                <div
                  className={cn(
                    "bg-muted p-4 rounded-md overflow-auto flex justify-center",
                    isExpanded ? "h-auto" : "max-h-[300px]",
                  )}
                >
                  <img
                    src={`data:image/png;base64,${result.plot}`}
                    alt="Plot"
                    className={cn("max-w-full h-auto", isExpanded ? "w-auto" : "max-h-[250px]")}
                  />
                </div>
              </TabsContent>
            )}
          </Tabs>
        ) : hasError ? (
          <div
            className="bg-red-50 dark:bg-red-950 p-4 rounded-md text-sm overflow-auto text-red-700 dark:text-red-300 font-mono whitespace-pre-wrap"
            style={{ maxHeight }}
          >
            {result.error}
          </div>
        ) : null}
      </div>
    </div>
  )
}

