"use client"
import Link from "next/link"
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
import { Microscope, Database, LineChart, Code, Home, BookOpen, Lightbulb, GraduationCap } from "lucide-react"

export function AppSidebar() {
  const pathname = usePathname()

  const algorithms = [
    {
      title: "Exploratory Data Analysis",
      slug: "eda",
      icon: Database,
    },
    {
      title: "Linear Regression",
      slug: "linear-regression",
      icon: LineChart,
    },
    {
      title: "Support Vector Machines",
      slug: "svm",
      icon: Code,
    },
  ]

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
                      <algorithm.icon className="h-4 w-4" />
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

