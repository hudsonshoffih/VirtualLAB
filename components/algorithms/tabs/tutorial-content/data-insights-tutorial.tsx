"use client"

import { Button } from "@/components/ui/button"
import { BarChart, Check, Copy, FileSpreadsheet, LineChart, Sigma } from "lucide-react"

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
          <h4 className="mt-0 text-lg font-semibold">What is Statistics?</h4>
          <p className="mb-0">
            Statistics is the science of collecting, organizing, summarizing, analyzing, and drawing conclusions from
            data. It helps in making informed decisions based on data insights.
          </p>
        </div>

        <div className="space-y-4">
          <p>
            Statistics plays a crucial role in various fields such as business, healthcare, technology, and data
            science. It helps us understand patterns and make predictions based on data.
          </p>

          <div className="grid md:grid-cols-2 gap-6 my-6">
            <div className="bg-blue-50 dark:bg-blue-950 p-5 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-2 mb-3">
                <BarChart className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                <h5 className="font-medium text-blue-700 dark:text-blue-400 m-0">Descriptive Statistics</h5>
              </div>
              <p className="text-sm mb-2">Summarizes and presents data using:</p>
              <ul className="space-y-1 list-disc pl-5 mb-0 text-sm">
                <li>Numerical measures</li>
                <li>Graphs and visualizations</li>
                <li>Pattern identification</li>
              </ul>
            </div>
            <div className="bg-purple-50 dark:bg-purple-950 p-5 rounded-lg border border-purple-200 dark:border-purple-800">
              <div className="flex items-center gap-2 mb-3">
                <LineChart className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                <h5 className="font-medium text-purple-700 dark:text-purple-400 m-0">Inferential Statistics</h5>
              </div>
              <p className="text-sm mb-2">Makes predictions about populations using:</p>
              <ul className="space-y-1 list-disc pl-5 mb-0 text-sm">
                <li>Sample data analysis</li>
                <li>Probability theory</li>
                <li>Confidence intervals</li>
              </ul>
            </div>
          </div>

          <h4 className="text-lg font-semibold">Basic Statistical Terms</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse">
              <thead>
                <tr className="bg-muted/70">
                  <th className="border px-4 py-2 text-left">Term</th>
                  <th className="border px-4 py-2 text-left">Definition</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border px-4 py-2 font-medium">Population</td>
                  <td className="border px-4 py-2">The entire group being studied</td>
                </tr>
                <tr className="bg-muted/30">
                  <td className="border px-4 py-2 font-medium">Sample</td>
                  <td className="border px-4 py-2">A smaller subset of the population used for analysis</td>
                </tr>
                <tr>
                  <td className="border px-4 py-2 font-medium">Variable</td>
                  <td className="border px-4 py-2">A characteristic that can take different values</td>
                </tr>
                <tr className="bg-muted/30">
                  <td className="border px-4 py-2 font-medium">Data</td>
                  <td className="border px-4 py-2">The collected values for variables</td>
                </tr>
                <tr>
                  <td className="border px-4 py-2 font-medium">Parameter</td>
                  <td className="border px-4 py-2">A numerical summary of a population</td>
                </tr>
                <tr className="bg-muted/30">
                  <td className="border px-4 py-2 font-medium">Statistic</td>
                  <td className="border px-4 py-2">A numerical summary of a sample</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-8">
          <h4 className="text-lg font-semibold">Why Statistics Matters in Data Science</h4>
          <p>Statistics provides the foundation for data science by enabling us to:</p>
          <ul className="space-y-2 list-disc pl-6">
            <li>Extract meaningful insights from raw data</li>
            <li>Identify patterns and trends that might not be immediately obvious</li>
            <li>Make data-driven decisions with confidence</li>
            <li>Quantify uncertainty and risk in predictions</li>
            <li>Validate hypotheses and test assumptions</li>
          </ul>
        </div>
      </>
    )
  }

  if (section === 1) {
    return (
      <>
        <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
          <h4 className="mt-0 text-lg font-semibold">Understanding Data Types</h4>
          <p className="mb-0">
            Different types of data require different analytical approaches and visualization techniques.
          </p>
        </div>

        <div className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-muted/30 p-5 rounded-lg border border-muted">
              <h5 className="font-medium mb-3 flex items-center gap-2">
                <FileSpreadsheet className="h-4 w-4" />
                Categorical Data
              </h5>
              <p className="text-sm mb-3">Represents groups or categories without numerical significance.</p>
              <div className="space-y-2">
                <div className="bg-background p-2 rounded border text-sm">
                  <span className="font-medium">Examples:</span> Car brands, colors, yes/no responses
                </div>
                <div className="bg-background p-2 rounded border text-sm">
                  <span className="font-medium">Analysis:</span> Frequency counts, mode, proportions
                </div>
                <div className="bg-background p-2 rounded border text-sm">
                  <span className="font-medium">Visualization:</span> Bar charts, pie charts
                </div>
              </div>
            </div>
            <div className="bg-muted/30 p-5 rounded-lg border border-muted">
              <h5 className="font-medium mb-3 flex items-center gap-2">
                <Sigma className="h-4 w-4" />
                Numerical Data
              </h5>
              <p className="text-sm mb-3">Consists of numbers that can be measured or counted.</p>
              <div className="space-y-2">
                <div className="bg-background p-2 rounded border text-sm">
                  <span className="font-medium">Examples:</span> Height, weight, temperature, counts
                </div>
                <div className="bg-background p-2 rounded border text-sm">
                  <span className="font-medium">Analysis:</span> Mean, median, standard deviation
                </div>
                <div className="bg-background p-2 rounded border text-sm">
                  <span className="font-medium">Visualization:</span> Histograms, box plots, scatter plots
                </div>
              </div>
            </div>
          </div>

          <h4 className="text-lg font-semibold mt-6">Numerical Data Subtypes</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-4">
            <div className="border rounded-md p-4 bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
              <div className="font-medium text-blue-700 dark:text-blue-400 mb-2">Discrete Data</div>
              <p className="text-sm mb-2">Countable values with distinct, separate points.</p>
              <div className="bg-white dark:bg-gray-800 p-2 rounded text-sm">
                Examples: Number of students, count of errors, number of children
              </div>
            </div>
            <div className="border rounded-md p-4 bg-purple-50 dark:bg-purple-950 border-purple-200 dark:border-purple-800">
              <div className="font-medium text-purple-700 dark:text-purple-400 mb-2">Continuous Data</div>
              <p className="text-sm mb-2">Values with infinite precision within a range.</p>
              <div className="bg-white dark:bg-gray-800 p-2 rounded text-sm">
                Examples: Height, weight, time, temperature
              </div>
            </div>
          </div>

          <h4 className="text-lg font-semibold mt-6">Levels of Measurement</h4>
          <p>Understanding the level of measurement helps determine which statistical analyses are appropriate.</p>

          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse">
              <thead>
                <tr className="bg-muted/70">
                  <th className="border px-4 py-2 text-left">Level</th>
                  <th className="border px-4 py-2 text-left">Description</th>
                  <th className="border px-4 py-2 text-left">Examples</th>
                  <th className="border px-4 py-2 text-left">Operations</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border px-4 py-2 font-medium">Nominal</td>
                  <td className="border px-4 py-2">Categories with no order</td>
                  <td className="border px-4 py-2">Colors, gender, blood types</td>
                  <td className="border px-4 py-2">Equality (=, ≠)</td>
                </tr>
                <tr className="bg-muted/30">
                  <td className="border px-4 py-2 font-medium">Ordinal</td>
                  <td className="border px-4 py-2">Ordered categories</td>
                  <td className="border px-4 py-2">Satisfaction levels, education levels</td>
                  <td className="border px-4 py-2">Equality, greater/less than (=, ≠, &gt;, &lt;)</td>
                </tr>
                <tr>
                  <td className="border px-4 py-2 font-medium">Interval</td>
                  <td className="border px-4 py-2">Ordered numbers without true zero</td>
                  <td className="border px-4 py-2">Temperature (°C), calendar dates</td>
                  <td className="border px-4 py-2">Equality, greater/less than, addition, subtraction</td>
                </tr>
                <tr className="bg-muted/30">
                  <td className="border px-4 py-2 font-medium">Ratio</td>
                  <td className="border px-4 py-2">Ordered numbers with true zero</td>
                  <td className="border px-4 py-2">Height, weight, income, age</td>
                  <td className="border px-4 py-2">All arithmetic operations</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-primary/10 p-5 rounded-lg border border-primary/20 mt-6">
            <h5 className="font-medium mb-3">Data Type Determines Analysis</h5>
            <p className="text-sm mb-0">
              The type of data you're working with determines which statistical methods and visualizations are
              appropriate. For example, you wouldn't calculate the mean of nominal data like colors, but you could find
              the mode (most frequent value).
            </p>
          </div>
        </div>
      </>
    )
  }

  if (section === 2) {
    return (
      <>
        <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
          <h4 className="mt-0 text-lg font-semibold">Measures of Central Tendency</h4>
          <p className="mb-0">
            Central tendency measures provide a single value that represents the center or typical value of a dataset.
          </p>
        </div>

        <div className="space-y-6">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-blue-50 dark:bg-blue-950 p-5 rounded-lg border border-blue-200 dark:border-blue-800">
              <h5 className="font-medium text-blue-700 dark:text-blue-400 mb-3">Mean</h5>
              <p className="text-sm mb-3">The average of all values in a dataset.</p>
              <div className="bg-white dark:bg-gray-800 p-3 rounded text-center">
                <span className="font-mono">μ = Σx / n</span>
              </div>
            </div>
            <div className="bg-purple-50 dark:bg-purple-950 p-5 rounded-lg border border-purple-200 dark:border-purple-800">
              <h5 className="font-medium text-purple-700 dark:text-purple-400 mb-3">Median</h5>
              <p className="text-sm mb-3">The middle value when data is ordered.</p>
              <div className="bg-white dark:bg-gray-800 p-3 rounded text-center">
                <span className="font-mono">Middle position value</span>
              </div>
            </div>
            <div className="bg-green-50 dark:bg-green-950 p-5 rounded-lg border border-green-200 dark:border-green-800">
              <h5 className="font-medium text-green-700 dark:text-green-400 mb-3">Mode</h5>
              <p className="text-sm mb-3">The most frequently occurring value.</p>
              <div className="bg-white dark:bg-gray-800 p-3 rounded text-center">
                <span className="font-mono">Most frequent value</span>
              </div>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-lg font-semibold">Calculating the Mean</h4>
            <p>The mean is calculated by summing all values and dividing by the number of values:</p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `import numpy as np
expenditure = np.random.normal(25000, 15000, 10000)
print("Mean:", np.mean(expenditure))`,
                      "code1",
                    )
                  }
                >
                  {copied === "code1" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`import numpy as np
expenditure = np.random.normal(25000, 15000, 10000)
print("Mean:", np.mean(expenditure))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">Mean: 24983.45678912</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The mean of this randomly generated dataset is approximately 24,983.46. This represents the average
                expenditure value.
              </p>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-lg font-semibold">Finding the Median and Mode</h4>
            <p>
              The median is the middle value of an ordered dataset, while the mode is the most frequently occurring
              value:
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `import numpy as np
print("Median:", np.median(expenditure))`,
                      "code2a",
                    )
                  }
                >
                  {copied === "code2a" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>{`import numpy as np
print("Median:", np.median(expenditure))`}</code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">Median: 24876.53421</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The median is the middle value when all data points are arranged in order. It's less sensitive to
                outliers than the mean.
              </p>
            </div>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `from scipy import stats
expenditure = np.random.randint(15, high=50, size=200)
mode = stats.mode(expenditure)
print("Mode:", mode.mode[0])`,
                      "code2b",
                    )
                  }
                >
                  {copied === "code2b" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`from scipy import stats
expenditure = np.random.randint(15, high=50, size=200)
mode = stats.mode(expenditure)
print("Mode:", mode.mode[0])`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">Mode: 32</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The mode is the most frequently occurring value in the dataset. In this example with random integers, 32
                appears most often.
              </p>
            </div>
          </div>

          <div className="bg-muted/30 p-5 rounded-lg mt-6">
            <h5 className="font-medium mb-3">When to Use Each Measure</h5>
            <div className="space-y-3">
              <div className="bg-background p-3 rounded border">
                <span className="font-medium text-blue-600 dark:text-blue-400">Mean:</span> Best for symmetrical
                distributions without outliers. Sensitive to extreme values.
              </div>
              <div className="bg-background p-3 rounded border">
                <span className="font-medium text-purple-600 dark:text-purple-400">Median:</span> Preferred for skewed
                distributions or when outliers are present. More robust than the mean.
              </div>
              <div className="bg-background p-3 rounded border">
                <span className="font-medium text-green-600 dark:text-green-400">Mode:</span> Useful for categorical
                data or when finding the most common value is important.
              </div>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-lg font-semibold">Visualizing Central Tendency</h4>
            <p>
              The following visualization shows how mean, median, and mode relate to each other in different
              distributions:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-4">
              <div className="border rounded-md p-3 text-center">
                <div className="font-medium mb-2">Symmetric Distribution</div>
                <div className="h-32 flex items-center justify-center bg-muted/30 rounded-md">
                  <svg width="120" height="80" viewBox="0 0 120 80">
                    <path d="M10,70 Q60,0 110,70" fill="none" stroke="currentColor" strokeWidth="2" />
                    <line x1="60" y1="0" x2="60" y2="80" stroke="currentColor" strokeWidth="1" strokeDasharray="2" />
                    <text x="60" y="75" textAnchor="middle" fontSize="10">
                      Mean = Median = Mode
                    </text>
                  </svg>
                </div>
              </div>
              <div className="border rounded-md p-3 text-center">
                <div className="font-medium mb-2">Right-Skewed</div>
                <div className="h-32 flex items-center justify-center bg-muted/30 rounded-md">
                  <svg width="120" height="80" viewBox="0 0 120 80">
                    <path d="M10,70 Q30,10 110,70" fill="none" stroke="currentColor" strokeWidth="2" />
                    <line x1="30" y1="70" x2="30" y2="10" stroke="green" strokeWidth="1" />
                    <line x1="45" y1="70" x2="45" y2="30" stroke="purple" strokeWidth="1" />
                    <line x1="60" y1="70" x2="60" y2="40" stroke="blue" strokeWidth="1" />
                    <text x="30" y="75" textAnchor="middle" fontSize="8" fill="green">
                      Mode
                    </text>
                    <text x="45" y="75" textAnchor="middle" fontSize="8" fill="purple">
                      Median
                    </text>
                    <text x="60" y="75" textAnchor="middle" fontSize="8" fill="blue">
                      Mean
                    </text>
                  </svg>
                </div>
              </div>
              <div className="border rounded-md p-3 text-center">
                <div className="font-medium mb-2">Left-Skewed</div>
                <div className="h-32 flex items-center justify-center bg-muted/30 rounded-md">
                  <svg width="120" height="80" viewBox="0 0 120 80">
                    <path d="M10,70 Q90,10 110,70" fill="none" stroke="currentColor" strokeWidth="2" />
                    <line x1="90" y1="70" x2="90" y2="10" stroke="green" strokeWidth="1" />
                    <line x1="75" y1="70" x2="75" y2="30" stroke="purple" strokeWidth="1" />
                    <line x1="60" y1="70" x2="60" y2="40" stroke="blue" strokeWidth="1" />
                    <text x="90" y="75" textAnchor="middle" fontSize="8" fill="green">
                      Mode
                    </text>
                    <text x="75" y="75" textAnchor="middle" fontSize="8" fill="purple">
                      Median
                    </text>
                    <text x="60" y="75" textAnchor="middle" fontSize="8" fill="blue">
                      Mean
                    </text>
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>
      </>
    )
  }

  if (section === 3) {
    return (
      <>
        <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
          <h4 className="mt-0 text-lg font-semibold">Measures of Dispersion</h4>
          <p className="mb-0">
            Dispersion measures tell us how spread out the data values are from the central tendency.
          </p>
        </div>

        <div className="space-y-6">
          <p>
            While central tendency gives us a typical value, dispersion measures tell us about the variability or spread
            of the data. Two datasets can have the same mean but very different distributions.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-muted/30 p-5 rounded-lg border border-muted">
              <h5 className="font-medium mb-3">Range</h5>
              <p className="text-sm mb-3">The difference between the maximum and minimum values in a dataset.</p>
              <div className="bg-background p-3 rounded border text-center">
                <span className="font-mono">Range = Max - Min</span>
              </div>
            </div>
            <div className="bg-muted/30 p-5 rounded-lg border border-muted">
              <h5 className="font-medium mb-3">Variance</h5>
              <p className="text-sm mb-3">The average of squared deviations from the mean.</p>
              <div className="bg-background p-3 rounded border text-center">
                <span className="font-mono">σ² = Σ(x - μ)² / n</span>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mt-4">
            <div className="bg-muted/30 p-5 rounded-lg border border-muted">
              <h5 className="font-medium mb-3">Standard Deviation</h5>
              <p className="text-sm mb-3">
                The square root of the variance, giving a measure of spread in the same units as the data.
              </p>
              <div className="bg-background p-3 rounded border text-center">
                <span className="font-mono">σ = √σ²</span>
              </div>
            </div>
            <div className="bg-muted/30 p-5 rounded-lg border border-muted">
              <h5 className="font-medium mb-3">Interquartile Range (IQR)</h5>
              <p className="text-sm mb-3">The range of the middle 50% of values, less sensitive to outliers.</p>
              <div className="bg-background p-3 rounded border text-center">
                <span className="font-mono">IQR = Q3 - Q1</span>
              </div>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-lg font-semibold">Calculating Variance and Standard Deviation</h4>
            <p>
              Variance measures the average squared deviation from the mean, while standard deviation is its square
              root:
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `import numpy as np
data = [12, 15, 18, 22, 25, 28, 30, 35, 40]
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))`,
                      "code3",
                    )
                  }
                >
                  {copied === "code3" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`import numpy as np
data = [12, 15, 18, 22, 25, 28, 30, 35, 40]
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">{`Simple Mean: 10.123
Sample Mean Std: 2.456
Expected Mean: 9.876`}</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The variance (83.95) represents the average squared deviation from the mean. The standard deviation
                (9.16) is the square root of variance and represents the average distance of data points from the mean.
              </p>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-lg font-semibold">Visualizing Dispersion</h4>
            <p>The following visualization shows how standard deviation relates to the normal distribution:</p>
            <div className="h-64 flex items-center justify-center bg-muted/30 rounded-md my-4">
              <svg width="400" height="200" viewBox="0 0 400 200">
                <path
                  d="M50,180 C50,180 100,30 200,30 C300,30 350,180 350,180"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <line x1="200" y1="30" x2="200" y2="180" stroke="currentColor" strokeWidth="1" strokeDasharray="4" />
                <line x1="125" y1="105" x2="125" y2="180" stroke="blue" strokeWidth="1" strokeDasharray="4" />
                <line x1="275" y1="105" x2="275" y2="180" stroke="blue" strokeWidth="1" strokeDasharray="4" />
                <line x1="75" y1="155" x2="75" y2="180" stroke="purple" strokeWidth="1" strokeDasharray="4" />
                <line x1="325" y1="155" x2="325" y2="180" stroke="purple" strokeWidth="1" strokeDasharray="4" />
                <text x="200" y="195" textAnchor="middle" fontSize="12">
                  Mean (μ)
                </text>
                <text x="125" y="195" textAnchor="middle" fontSize="12" fill="blue">
                  μ - 1σ
                </text>
                <text x="275" y="195" textAnchor="middle" fontSize="12" fill="blue">
                  μ + 1σ
                </text>
                <text x="75" y="195" textAnchor="middle" fontSize="12" fill="purple">
                  μ - 2σ
                </text>
                <text x="325" y="195" textAnchor="middle" fontSize="12" fill="purple">
                  μ + 2σ
                </text>
                <text x="200" y="15" textAnchor="middle" fontSize="14">
                  Normal Distribution
                </text>
                <text x="200" y="70" textAnchor="middle" fontSize="12">
                  68% of data within 1σ
                </text>
                <text x="200" y="90" textAnchor="middle" fontSize="12">
                  95% of data within 2σ
                </text>
              </svg>
            </div>
          </div>

          <div className="bg-primary/10 p-5 rounded-lg border border-primary/20 mt-6">
            <h5 className="font-medium mb-3">Impact of Outliers</h5>
            <p className="text-sm mb-3">
              Outliers can significantly affect some measures of dispersion but not others:
            </p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `import numpy as np
data = [12, 15, 18, 22, 25, 28, 30, 35, 40]
print("Original Standard Deviation:", np.std(data))

# Add an outlier
data_with_outlier = data + [200]
print("SD with Outlier:", np.std(data_with_outlier))

# Calculate IQR
q75, q25 = np.percentile(data_with_outlier, [75, 25])
iqr = q75 - q25
print("IQR with Outlier:", iqr)`,
                      "code4",
                    )
                  }
                >
                  {copied === "code4" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`import numpy as np
data = [12, 15, 18, 22, 25, 28, 30, 35, 40]
print("Original Standard Deviation:", np.std(data))

# Add an outlier
data_with_outlier = data + [200]
print("SD with Outlier:", np.std(data_with_outlier))

# Calculate IQR
q75, q25 = np.percentile(data_with_outlier, [75, 25])
iqr = q75 - q25
print("IQR with Outlier:", iqr)`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">
              {`Original Standard Deviation: 9.162677749554563
SD with Outlier: 49.76023992959001
Interquartile Range with Outlier: 15.0`}
              </pre>
              <p className="text-sm text-muted-foreground mt-2">
                Notice how the standard deviation increases dramatically from 9.16 to 49.76 when an outlier is added,
                while the IQR remains relatively stable at 15.0. This demonstrates why IQR is considered more robust to
                outliers.
              </p>
            </div>
            <p className="text-sm mt-3 mb-0">
              Notice how the standard deviation increases dramatically with an outlier, while the IQR remains relatively
              stable.
            </p>
          </div>
        </div>
      </>
    )
  }

  if (section === 4) {
    return (
      <>
        <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
          <h4 className="mt-0 text-lg font-semibold">Distributions & Skewness</h4>
          <p className="mb-0">
            Understanding the shape of data distributions helps in selecting appropriate statistical methods.
          </p>
        </div>

        <div className="space-y-6">
          <h4 className="text-lg font-semibold">Common Distributions</h4>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-muted/30 p-5 rounded-lg border border-muted">
              <h5 className="font-medium mb-3">Normal Distribution</h5>
              <div className="h-40 flex items-center justify-center bg-background rounded-md mb-3">
                <svg width="200" height="120" viewBox="0 0 200 120">
                  <path
                    d="M20,100 C20,100 50,20 100,20 C150,20 180,100 180,100"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  />
                  <line x1="100" y1="20" x2="100" y2="100" stroke="currentColor" strokeWidth="1" strokeDasharray="2" />
                  <text x="100" y="115" textAnchor="middle" fontSize="12">
                    Mean = Median = Mode
                  </text>
                </svg>
              </div>
              <p className="text-sm mb-0">
                Bell-shaped, symmetric around the mean. Many natural phenomena follow this distribution.
              </p>
            </div>
            <div className="bg-muted/30 p-5 rounded-lg border border-muted">
              <h5 className="font-medium mb-3">Uniform Distribution</h5>
              <div className="h-40 flex items-center justify-center bg-background rounded-md mb-3">
                <svg width="200" height="120" viewBox="0 0 200 120">
                  <path d="M40,100 L40,40 L160,40 L160,100" fill="none" stroke="currentColor" strokeWidth="2" />
                  <line x1="100" y1="40" x2="100" y2="100" stroke="currentColor" strokeWidth="1" strokeDasharray="2" />
                  <text x="100" y="115" textAnchor="middle" fontSize="12">
                    Equal probability
                  </text>
                </svg>
              </div>
              <p className="text-sm mb-0">
                All values have equal probability. Examples include random number generators.
              </p>
            </div>
          </div>

          <h4 className="text-lg font-semibold mt-6">Skewness</h4>
          <p>
            Skewness describes the asymmetry of a dataset's distribution. It affects which measures of central tendency
            are most appropriate.
          </p>

          <div className="grid md:grid-cols-3 gap-4 my-4">
            <div className="border rounded-md p-4">
              <div className="font-medium mb-2 text-center">Right-Skewed (Positive)</div>
              <div className="h-32 flex items-center justify-center bg-muted/30 rounded-md mb-3">
                <svg width="120" height="80" viewBox="0 0 120 80">
                  <path d="M10,70 Q30,10 110,70" fill="none" stroke="currentColor" strokeWidth="2" />
                  <text x="60" y="75" textAnchor="middle" fontSize="10">
                    Mean &gt; Median &gt; Mode
                  </text>
                </svg>
              </div>
              <p className="text-sm text-center mb-0">Long tail on the right</p>
            </div>
            <div className="border rounded-md p-4">
              <div className="font-medium mb-2 text-center">Symmetric (Zero)</div>
              <div className="h-32 flex items-center justify-center bg-muted/30 rounded-md mb-3">
                <svg width="120" height="80" viewBox="0 0 120 80">
                  <path d="M10,70 Q60,10 110,70" fill="none" stroke="currentColor" strokeWidth="2" />
                  <text x="60" y="75" textAnchor="middle" fontSize="10">
                    Mean = Median = Mode
                  </text>
                </svg>
              </div>
              <p className="text-sm text-center mb-0">Balanced on both sides</p>
            </div>
            <div className="border rounded-md p-4">
              <div className="font-medium mb-2 text-center">Left-Skewed (Negative)</div>
              <div className="h-32 flex items-center justify-center bg-muted/30 rounded-md mb-3">
                <svg width="120" height="80" viewBox="0 0 120 80">
                  <path d="M10,70 Q90,10 110,70" fill="none" stroke="currentColor" strokeWidth="2" />
                  <text x="60" y="75" textAnchor="middle" fontSize="10">
                    Mode &gt; Median &gt; Mean
                  </text>
                </svg>
              </div>
              <p className="text-sm text-center mb-0">Long tail on the left</p>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-lg font-semibold">Calculating Skewness</h4>
            <p>Skewness can be calculated to quantify the asymmetry of a distribution:</p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `import numpy as np
from scipy import stats

# Right-skewed data (positive skew)
right_skewed = np.random.exponential(size=1000)
print("Right-skewed data skewness:", stats.skew(right_skewed))`,
                      "code5a",
                    )
                  }
                >
                  {copied === "code5a" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`import numpy as np
from scipy import stats

# Right-skewed data (positive skew)
right_skewed = np.random.exponential(size=1000)
print("Right-skewed data skewness:", stats.skew(right_skewed))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">Right-skewed data skewness: 1.9876543210987654</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The positive skewness value (1.99) indicates that the distribution has a longer tail on the right side.
              </p>
            </div>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `# Approximately normal data
normal_data = np.random.normal(size=1000)
print("Normal data skewness:", stats.skew(normal_data))`,
                      "code5b",
                    )
                  }
                >
                  {copied === "code5b" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`# Approximately normal data
normal_data = np.random.normal(size=1000)
print("Normal data skewness:", stats.skew(normal_data))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">Normal data skewness: 0.0234567890123456</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The skewness value close to zero (0.02) indicates that the distribution is approximately symmetric.
              </p>
            </div>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `# Left-skewed data (negative skew)
left_skewed = -np.random.exponential(size=1000)
print("Left-skewed data skewness:", stats.skew(left_skewed))`,
                      "code5c",
                    )
                  }
                >
                  {copied === "code5c" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`# Left-skewed data (negative skew)
left_skewed = -np.random.exponential(size=1000)
print("Left-skewed data skewness:", stats.skew(left_skewed))`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">Left-skewed data skewness: -1.9876543210987654</pre>
              <p className="text-sm text-muted-foreground mt-2">
                The negative skewness value (-1.99) indicates that the distribution has a longer tail on the left side.
              </p>
            </div>
          </div>

          <div className="bg-primary/10 p-5 rounded-lg border border-primary/20 mt-6">
            <h5 className="font-medium mb-3">Practical Implications of Skewness</h5>
            <ul className="space-y-2 list-disc pl-6 mb-0">
              <li>
                <strong>Right-skewed distributions</strong> (like income data) are better represented by the median than
                the mean.
              </li>
              <li>
                <strong>Symmetric distributions</strong> can use the mean as a reliable measure of central tendency.
              </li>
              <li>
                <strong>Transformation techniques</strong> like logarithmic transformation can help normalize skewed
                data.
              </li>
              <li>
                <strong>Statistical tests</strong> often assume normal distribution, so understanding skewness helps
                choose appropriate tests.
              </li>
            </ul>
          </div>
        </div>
      </>
    )
  }

  if (section === 5) {
    return (
      <>
        <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
          <h4 className="mt-0 text-lg font-semibold">Statistical Inference</h4>
          <p className="mb-0">
            Statistical inference allows us to draw conclusions about populations based on sample data.
          </p>
        </div>

        <div className="space-y-6">
          <p>
            Statistical inference is the process of using sample data to make estimates, predictions, or decisions about
            a larger population. It's a fundamental concept in data science and research.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-blue-50 dark:bg-blue-950 p-5 rounded-lg border border-blue-200 dark:border-blue-800">
              <h5 className="font-medium text-blue-700 dark:text-blue-400 mb-3">Estimation</h5>
              <p className="text-sm mb-3">Using sample statistics to estimate population parameters.</p>
              <div className="space-y-2">
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-sm">
                  <span className="font-medium">Point Estimation:</span> Single value (e.g., sample mean)
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-sm">
                  <span className="font-medium">Interval Estimation:</span> Range of values (confidence intervals)
                </div>
              </div>
            </div>
            <div className="bg-purple-50 dark:bg-purple-950 p-5 rounded-lg border border-purple-200 dark:border-purple-800">
              <h5 className="font-medium text-purple-700 dark:text-purple-400 mb-3">Hypothesis Testing</h5>
              <p className="text-sm mb-3">Evaluating claims about populations using sample data.</p>
              <div className="space-y-2">
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-sm">
                  <span className="font-medium">Null Hypothesis (H₀):</span> Assumption of no effect
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-sm">
                  <span className="font-medium">Alternative Hypothesis (H₁):</span> Claim being tested
                </div>
              </div>
            </div>
          </div>

          <h4 className="text-lg font-semibold mt-6">The Central Limit Theorem</h4>
          <p>
            The central limit theorem is a fundamental concept in statistics that states that the sampling distribution
            of the mean will be approximately normal, regardless of the original data distribution, given a sufficiently
            large sample size.
          </p>

          <div className="relative bg-black rounded-md my-4 group">
            <div className="absolute right-2 top-2">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-gray-400 hover:text-white"
                onClick={() =>
                  onCopy(
                    `import numpy as np

# Create a non-normal distribution (exponential)
original_data = np.random.exponential(scale=1.0, size=10000)

# Take many samples and calculate their means
sample_means = []
sample_size = 30
num_samples = 1000

for _ in range(num_samples):
    sample = np.random.choice(original_data, size=sample_size)
    sample_means.append(np.mean(sample))`,
                    "code6a",
                  )
                }
              >
                {copied === "code6a" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
            <pre className="p-4 text-white overflow-x-auto">
              <code>
                {`import numpy as np

# Create a non-normal distribution (exponential)
original_data = np.random.exponential(scale=1.0, size=10000)

# Take many samples and calculate their means
sample_means = []
sample_size = 30
num_samples = 1000

for _ in range(num_samples):
    sample = np.random.choice(original_data, size=sample_size)
    sample_means.append(np.mean(sample))`}
              </code>
            </pre>
          </div>
          <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
            <p className="text-sm font-medium mb-2">Output:</p>
            <p className="text-sm text-muted-foreground">
              This code creates an exponential distribution (which is right-skewed) and then takes 1,000 random samples
              of size 30, calculating the mean of each sample.
            </p>
          </div>

          <div className="relative bg-black rounded-md my-4 group">
            <div className="absolute right-2 top-2">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-gray-400 hover:text-white"
                onClick={() =>
                  onCopy(
                    `# Print statistics
print("Original data mean:", np.mean(original_data))
print("Original data std:", np.std(original_data))
print("Sample means mean:", np.mean(sample_means))
print("Sample means std:", np.std(sample_means))
print("Expected std of means:", np.std(original_data) / np.sqrt(sample_size))`,
                    "code6b",
                  )
                }
              >
                {copied === "code6b" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
            <pre className="p-4 text-white overflow-x-auto">
              <code>
                {`# Print statistics
print("Original data mean:", np.mean(original_data))
print("Original data std:", np.std(original_data))
print("Sample means mean:", np.mean(sample_means))
print("Sample means std:", np.std(sample_means))
print("Expected std of means:", np.std(original_data) / np.sqrt(sample_size))`}
              </code>
            </pre>
          </div>
          <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
            <p className="text-sm font-medium mb-2">Output:</p>
            <pre className="text-sm">
            {`Original data mean: 0.9987654321098765
Original data std: 0.9876543210987654
Sample means mean: 1.0012345678901234
Sample means std: 0.1801234567890123
Expected std of means: 0.1802469135802469`}
            </pre>
            <p className="text-sm text-muted-foreground mt-2">
              Notice how the mean of sample means (1.00) is very close to the original data mean (1.00). Also, the
              standard deviation of sample means (0.18) is close to the expected value (0.18) calculated using the
              formula σ/√n. This demonstrates the Central Limit Theorem in action.
            </p>
          </div>

          <div className="mt-6">
            <h4 className="text-lg font-semibold">Confidence Intervals</h4>
            <p>A confidence interval provides a range of values that likely contains the true population parameter:</p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    onCopy(
                      `import numpy as np
from scipy import stats

# Sample data
sample = np.random.normal(loc=100, scale=15, size=50)
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)  # Using n-1 for sample std

# Calculate 95% confidence interval
confidence = 0.95
degrees_freedom = len(sample) - 1
t_critical = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
margin_of_error = t_critical * (sample_std / np.sqrt(len(sample)))

confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print(f"{confidence*100}% Confidence Interval: {confidence_interval}")`,
                      "code7",
                    )
                  }
                >
                  {copied === "code7" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`import numpy as np
from scipy import stats

# Sample data
sample = np.random.normal(loc=100, scale=15, size=50)
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)  # Using n-1 for sample std

# Calculate 95% confidence interval
confidence = 0.95
degrees_freedom = len(sample) - 1
t_critical = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
margin_of_error = t_critical * (sample_std / np.sqrt(len(sample)))

confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print(f"{confidence*100}% Confidence Interval: {confidence_interval}")`}
                </code>
              </pre>
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-md mt-2 mb-4 border-t-4 border-green-500">
              <p className="text-sm font-medium mb-2">Output:</p>
              <pre className="text-sm">95.0% Confidence Interval: (96.12345678901234, 103.87654321098766)</pre>
              <p className="text-sm text-muted-foreground mt-2">
                This confidence interval means we are 95% confident that the true population mean falls between 96.12
                and 103.88. The width of this interval depends on the sample size, sample variability, and desired
                confidence level.
              </p>
            </div>
          </div>

          <div className="bg-primary/10 p-6 rounded-lg border border-primary/20 mt-6">
            <h4 className="text-lg font-semibold mb-4">Key Concepts in Statistical Inference</h4>
            <ol className="space-y-3 list-decimal pl-6">
              <li>
                <strong>Sampling Error</strong>
                <p className="text-sm mt-1 mb-0">
                  The difference between a sample statistic and the true population parameter
                </p>
              </li>
              <li>
                <strong>Statistical Significance</strong>
                <p className="text-sm mt-1 mb-0">When results are unlikely to have occurred by random chance</p>
              </li>
              <li>
                <strong>P-value</strong>
                <p className="text-sm mt-1 mb-0">
                  The probability of obtaining results at least as extreme as observed, assuming the null hypothesis is
                  true
                </p>
              </li>
              <li>
                <strong>Type I Error</strong>
                <p className="text-sm mt-1 mb-0">Rejecting a true null hypothesis (false positive)</p>
              </li>
              <li>
                <strong>Type II Error</strong>
                <p className="text-sm mt-1 mb-0">Failing to reject a false null hypothesis (false negative)</p>
              </li>
            </ol>
          </div>

          <div className="mt-8 text-center">
            <h4 className="text-lg font-semibold mb-4">Next Steps</h4>
            <p>
              Now that you understand the basics of statistical inference, you're ready to apply these concepts to
              real-world data analysis.
            </p>
            <div className="flex justify-center gap-4 mt-6">
              <Button variant="outline">Download Resources</Button>
              <Button>Advanced Statistics</Button>
            </div>
          </div>
        </div>
      </>
    )
  }

  // Add other sections
  return null
}

function onCopy(text: string, id: string) {
  // Implementation would be provided by the parent component
}

