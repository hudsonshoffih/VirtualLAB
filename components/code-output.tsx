"use client"

import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

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
}

export function CodeOutput({ result, isExecuting }: CodeOutputProps) {
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
  if (hasOutput) tabs.push("console")
  if (hasTable) tabs.push("table")
  if (hasPlot) tabs.push("plot")

  return (
    <div className="space-y-2">
      {hasOutput || hasTable || hasPlot ? (
        <Tabs defaultValue={tabs[0]} className="w-full">
          <TabsList className="grid w-full" style={{ gridTemplateColumns: `repeat(${tabs.length}, 1fr)` }}>
            {hasOutput && <TabsTrigger value="console">Console</TabsTrigger>}
            {hasTable && <TabsTrigger value="table">Table</TabsTrigger>}
            {hasPlot && <TabsTrigger value="plot">Plot</TabsTrigger>}
          </TabsList>

          {hasOutput && (
            <TabsContent value="console" className="mt-2">
              <pre className="bg-muted p-2 rounded-md text-xs overflow-auto max-h-[300px] whitespace-pre-wrap">
                {result.output}
              </pre>
            </TabsContent>
          )}

          {hasTable && (
            <TabsContent value="table" className="mt-2">
              <div
                className="bg-muted p-2 rounded-md text-xs overflow-auto max-h-[300px]"
                dangerouslySetInnerHTML={{ __html: result.table_html || "" }}
              />
            </TabsContent>
          )}

          {hasPlot && (
            <TabsContent value="plot" className="mt-2">
              <div className="bg-muted p-2 rounded-md overflow-auto max-h-[300px] flex justify-center">
                <img src={`data:image/png;base64,${result.plot}`} alt="Plot" className="max-w-full h-auto" />
              </div>
            </TabsContent>
          )}
        </Tabs>
      ) : hasError ? (
        <div className="bg-red-50 dark:bg-red-950 p-2 rounded-md text-xs overflow-auto max-h-[300px] text-red-700 dark:text-red-300 whitespace-pre-wrap">
          {result.error}
        </div>
      ) : null}
    </div>
  )
}

