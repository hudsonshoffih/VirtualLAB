"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { Algorithm } from "@/lib/types"
import { TutorialTab } from "@/components/algorithms/tabs/tutorial-tab"
import { DemoTab } from "@/components/algorithms/tabs/demo-tab"
import { PracticeTab } from "@/components/algorithms/tabs/practice-tab"
import { TestTab } from "@/components/algorithms/tabs/test-tab"

interface AlgorithmTabsProps {
  algorithm: Algorithm
}

export function AlgorithmTabs({ algorithm }: AlgorithmTabsProps) {
  const [activeTab, setActiveTab] = useState("tutorial")

  return (
    <Tabs defaultValue="tutorial" value={activeTab} onValueChange={setActiveTab} className="w-full">
      <TabsList className="grid w-full grid-cols-4">
        <TabsTrigger value="tutorial">Tutorial</TabsTrigger>
        <TabsTrigger value="demo">Demo</TabsTrigger>
        <TabsTrigger value="practice">Practice</TabsTrigger>
        <TabsTrigger value="test">Test</TabsTrigger>
      </TabsList>
      <TabsContent value="tutorial">
        <TutorialTab algorithm={algorithm} />
      </TabsContent>
      <TabsContent value="demo">
        <DemoTab algorithm={algorithm} />
      </TabsContent>
      <TabsContent value="practice">
        <PracticeTab algorithm={algorithm} />
      </TabsContent>
      <TabsContent value="test">
        <TestTab algorithm={algorithm} />
      </TabsContent>
    </Tabs>
  )
}

