"use client"

import { useSearchParams } from "next/navigation"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TutorialTab } from "./tabs/tutorial-tab"
import { PracticeTab } from "./tabs/practice-tab"
import { TestTab } from "./tabs/test-tab"
import { DemoTab } from "./tabs/demo-tab"
import type { Algorithm } from "@/lib/types"

interface AlgorithmTabsProps {
  algorithm: Algorithm
}

export function AlgorithmTabs({ algorithm }: AlgorithmTabsProps) {
  const searchParams = useSearchParams()
  const defaultTab = searchParams.get("tab") || "tutorial"

  return (
    <Tabs defaultValue={defaultTab} className="w-full">
      <TabsList className="grid w-full grid-cols-4">
        <TabsTrigger value="tutorial">Tutorial</TabsTrigger>
        <TabsTrigger value="practice">Practice</TabsTrigger>
        <TabsTrigger value="demo">Demo</TabsTrigger>
        <TabsTrigger value="test">Quiz</TabsTrigger>
      </TabsList>

      <TabsContent value="tutorial" className="mt-6">
        <TutorialTab algorithm={algorithm} />
      </TabsContent>

      <TabsContent value="practice" className="mt-6">
        <PracticeTab algorithm={algorithm} />
      </TabsContent>

      <TabsContent value="demo" className="mt-6">
        <DemoTab algorithm={algorithm} />
      </TabsContent>

      <TabsContent value="test" className="mt-6">
        <TestTab algorithm={algorithm} />
      </TabsContent>
    </Tabs>
  )
}
