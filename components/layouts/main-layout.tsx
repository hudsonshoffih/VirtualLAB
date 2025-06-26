"use client"

import React from "react"
import { Sidebar, SidebarInset } from "@/components/ui/sidebar"
import { ModeToggle } from "@/components/mode-toggle"
import { Button } from "@/components/ui/button"
import {
  Menu,
  Microscope,
  Home,
  BookOpen,
  Lightbulb,
  GraduationCap,
  Database,
  LineChart,
  Code,
  BarChart,
  PieChart,
  GitBranch,
  Network,
  Layers,
  CircleDot,
  Boxes,
} from "lucide-react"
import { useSidebar } from "@/components/ui/sidebar"
import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
} from "@/components/ui/sidebar"
import { getAlgorithms } from "@/lib/algorithms"
import { CommandSearch } from "@/components/search/command-search"

export function MainLayout({ children }: { children: React.ReactNode }) {
  const { toggleSidebar } = useSidebar()
  const pathname = usePathname()
  const algorithms = getAlgorithms()

  return (
    <div className="flex min-h-screen">
      <MainSidebar pathname={pathname} />
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-16 items-center gap-4 border-b bg-background px-6">
          <Button variant="ghost" size="icon" onClick={toggleSidebar} className="md:hidden">
            <Menu className="h-5 w-5" />
            <span className="sr-only">Toggle menu</span>
          </Button>
          <div className="ml-auto flex items-center gap-2">
            <CommandSearch />
            <ModeToggle />
          </div>
        </header>
        <main className="flex-1">{children}</main>
      </SidebarInset>
    </div>
  )
}

function MainSidebar({ pathname }: { pathname: string }) {
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
          <SidebarGroupLabel>Resources</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <Link href="/resources/tutorials">
                    <BookOpen className="h-4 w-4" />
                    <span>Tutorials</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <Link href="/resources/examples">
                    <Lightbulb className="h-4 w-4" />
                    <span>Examples</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <Link href="/resources/courses">
                    <GraduationCap className="h-4 w-4" />
                    <span>Courses</span>
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
