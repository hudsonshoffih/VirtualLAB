"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BookOpen, Code, BarChart } from "lucide-react"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartTooltipItem,
  ChartBar,
  ChartBarItem,
  ChartTitle,
} from "@/components/ui/chart"

export function VisualizationTab() {
  const [activeTab, setActiveTab] = useState("explanation")

  // Example feature importance data
  const featureImportanceData = [
    { parameter: "Feature A", importance: 0.85 },
    { parameter: "Feature B", importance: 0.62 },
    { parameter: "Feature C", importance: 0.45 },
    { parameter: "Feature D", importance: 0.38 },
    { parameter: "Feature E", importance: 0.22 },
  ]

  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="explanation">
            <BookOpen className="h-4 w-4 mr-2" />
            Explanation
          </TabsTrigger>
          <TabsTrigger value="code">
            <Code className="h-4 w-4 mr-2" />
            Code
          </TabsTrigger>
          <TabsTrigger value="visualization">
            <BarChart className="h-4 w-4 mr-2" />
            Visualization
          </TabsTrigger>
        </TabsList>

        <TabsContent value="visualization" className="mt-4">
          <Card className="p-5">
            <h4 className="font-medium text-lg mb-3">Feature Importance</h4>
            <div className="h-80">
              <ChartContainer>
                <ChartTitle>Feature Importance</ChartTitle>
                <ChartBar data={featureImportanceData}>
                  {(data) => (
                    <>
                      <ChartBarItem
                        data={data}
                        valueAccessor={(d) => d.importance}
                        categoryAccessor={(d) => d.parameter}
                        style={{ fill: "#6366f1" }}
                      />
                      <ChartTooltip>
                        {({ point }) => (
                          <ChartTooltipContent>
                            <ChartTooltipItem label="Feature" value={point.data.parameter} />
                            <ChartTooltipItem label="Importance" value={point.data.importance.toFixed(2)} />
                          </ChartTooltipContent>
                        )}
                      </ChartTooltip>
                    </>
                  )}
                </ChartBar>
              </ChartContainer>
            </div>
            <div className="mt-2 text-sm text-muted-foreground">
              <p>
                The chart shows the relative importance of each feature in the model. Features are ranked by their
                contribution to the model's predictions, with higher values indicating more important features.
              </p>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

