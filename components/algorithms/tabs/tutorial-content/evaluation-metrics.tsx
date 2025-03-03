import { Button } from "@/components/ui/button"
import { BarChart, Check, Copy, FileSpreadsheet, LineChart, Sigma } from "lucide-react"

interface EvaluationMetricsProps {
    section: number
    onCopy: (text: string, id: string) => void
    copied: string | null
  }
  
  export function EvaluationMetrics({ section, onCopy, copied }: EvaluationMetricsProps) {
  }
  //     <Card className="space-y-4">