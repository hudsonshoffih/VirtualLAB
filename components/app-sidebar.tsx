"use client"
import Link from "next/link"
import React from "react"

import { usePathname } from "next/navigation"
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
} from "@/components/ui/sidebar"
import {
  Microscope,
  Database,
  LineChart,
  Code,
  Home,
  BookOpen,
  Lightbulb,
  GraduationCap,
  BarChart,
  PieChart,
  GitBranch,
  Network,
  Layers,
  CircleDot,
  Boxes,
} from "lucide-react"
import { getAlgorithms } from "@/lib/algorithms"

export function AppSidebar() {
  const pathname = usePathname()
  const algorithms = getAlgorithms()

  // Map icons to algorithm slugs
  const algorithmIcons: Record<string, any> = {
    eda: Database,
    "dataset-insights": BarChart,
    "evaluation-metrics": PieChart,
    "linear-regression": LineChart,
    "logistic-regression": LineChart,
    knn: Network,
    "random-forest": GitBranch,
    svm: Code,
    "ensemble-models": Layers,
    kmeans: CircleDot,
    pca: Boxes,
  }

  return (
    <Sidebar>
      <SidebarHeader className="border-b">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild size="lg">
              <Link href="/" className="flex items-center gap-2">
                <Microscope className="h-5 w-5" />
                <span className="font-semibold">Virtual Lab</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild isActive={pathname === "/"}>
              <Link href="/">
                <Home className="h-4 w-4" />
                <span>Home</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>

        <SidebarGroup>
          <SidebarGroupLabel>Algorithms</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {algorithms.map((algorithm) => (
                <SidebarMenuItem key={algorithm.slug}>
                  <SidebarMenuButton asChild isActive={pathname === `/algorithms/${algorithm.slug}`}>
                    <Link href={`/algorithms/${algorithm.slug}`}>
                      {algorithmIcons[algorithm.slug] ? (
                        React.createElement(algorithmIcons[algorithm.slug], { className: "h-4 w-4" })
                      ) : (
                        <Code className="h-4 w-4" />
                      )}
                      <span>{algorithm.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>About Virtual Lab</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <Link href="/resources/tutorials">
                    <BookOpen className="h-4 w-4" />
                    <span>Contributors</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <Link href="/resources/examples">
                    <Lightbulb className="h-4 w-4" />
                    <span>Feedback</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <Link href="/resources/courses">
                    <GraduationCap className="h-4 w-4" />
                    <span>Reference</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  )
}

