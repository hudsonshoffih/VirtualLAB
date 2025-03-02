interface DataInsightsTutorialProps {
    section: number
    onCopy: (text: string, id: string) => void
    copied: string | null
  }
  
  export function DataInsightsTutorial({ section, onCopy, copied }: DataInsightsTutorialProps) {
    if (section === 0) {
      return (
        <>
          <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
            <h4 className="mt-0 text-lg font-semibold">Understanding Data Insights</h4>
            <p className="mb-0">
              Data insights help us understand patterns, trends, and relationships within our datasets.
            </p>
          </div>
          {/* Add specific content for Data Insights */}
        </>
      )
    }
  
    // Add other sections
    return null
  }
  
  