import { cn } from "@/lib/utils"
import * as React from "react"

const Code = React.forwardRef<HTMLPreElement, React.HTMLAttributes<HTMLPreElement>>(({ className, ...props }, ref) => {
  return (
    <pre
      className={cn("relative rounded-md border bg-muted py-4 font-mono text-sm font-semibold", className)}
      ref={ref}
      {...props}
    />
  )
})
Code.displayName = "Code"

export { Code }

