"use client"

import type React from "react"

import { cn } from "@/lib/utils"
import { motion, AnimatePresence } from "framer-motion"
import Link from "next/link"
import { useState } from "react"

interface DockItem {
  title: string
  icon: React.ComponentType<{ className?: string }>
  href: string
}

interface FloatingDockProps {
  items: DockItem[]
  className?: string
}

export function FloatingDock({ items, className }: FloatingDockProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  return (
    <div className={cn("relative", className)}>
      <motion.div
        className="flex items-end gap-2 p-3 bg-background/80 backdrop-blur-md border border-border/50 rounded-2xl shadow-2xl"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {items.map((item, index) => {
          const Icon = item.icon
          return (
            <div key={item.title} className="relative">
              <AnimatePresence>
                {hoveredIndex === index && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.6 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 2, scale: 0.6 }}
                    className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-1 bg-foreground text-background text-sm rounded-lg whitespace-nowrap z-50"
                  >
                    {item.title}
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-foreground" />
                  </motion.div>
                )}
              </AnimatePresence>

              <Link href={item.href}>
                <motion.div
                  className="relative flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-primary/10 to-primary/5 hover:from-primary/20 hover:to-primary/10 border border-primary/20 hover:border-primary/40 transition-all duration-300 cursor-pointer"
                  onMouseEnter={() => setHoveredIndex(index)}
                  onMouseLeave={() => setHoveredIndex(null)}
                  whileHover={{
                    scale: 1.2,
                    y: -8,
                    transition: { duration: 0.2 },
                  }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Icon className="h-6 w-6 text-primary" />

                  {/* Glow effect */}
                  <motion.div
                    className="absolute inset-0 rounded-xl bg-primary/20 blur-md -z-10"
                    initial={{ opacity: 0 }}
                    whileHover={{ opacity: 1 }}
                    transition={{ duration: 0.2 }}
                  />
                </motion.div>
              </Link>
            </div>
          )
        })}
      </motion.div>

      {/* Base glow */}
      <motion.div
        className="absolute inset-0 bg-primary/10 rounded-2xl blur-xl -z-10"
        animate={{
          scale: [1, 1.05, 1],
          opacity: [0.3, 0.6, 0.3],
        }}
        transition={{
          duration: 3,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
      />
    </div>
  )
}
