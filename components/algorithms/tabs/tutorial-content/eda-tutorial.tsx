import { Button } from "@/components/ui/button"
import { Check, Copy } from "lucide-react"

interface EdaTutorialProps {
    section: number
    onCopy: (text: string, id: string) => void
    copied: string | null
  }
  
  export function EdaTutorial({ section, onCopy, copied }: EdaTutorialProps) {
    if (section === 0) {
        return (
        <>
          <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
            <h4 className="mt-0 text-lg font-semibold">What is EDA?</h4>
            <p className="mb-0">
              Exploratory Data Analysis (EDA) is a fundamental step in data science that involves
              summarizing and visualizing data to extract meaningful insights.
            </p>
          </div>

          <div className="space-y-4">
            <p>
              EDA helps in identifying patterns, detecting anomalies, and understanding relationships
              between different variables. Before applying machine learning models, EDA allows us to:
            </p>

            <ul className="space-y-2 list-disc pl-6">
              <li>Assess the quality of the dataset</li>
              <li>Make necessary adjustments (handling missing values, transforming variables)</li>
              <li>Identify missing data and detect outliers</li>
              <li>Visualize distributions</li>
              <li>Define problem statements for further analysis</li>
            </ul>
          </div>

          <div className="mt-8">
            <h4 className="text-lg font-semibold">How to Perform EDA?</h4>
            <p>
              The approach to EDA varies depending on the dataset being analyzed. In this tutorial, we will
              perform EDA on the <strong>Iris Dataset</strong>, a simple dataset widely used for
              classification problems in machine learning.
            </p>
            <p>
              The dataset consists of measurements of different iris flower species, allowing us to explore
              patterns and relationships between features.
            </p>
          </div>
        </>
      )}
  
    if (section === 1) {
        return (
            <>
                <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
                    <h4 className="mt-0 text-lg font-semibold">The Iris Dataset</h4>
                    <p className="mb-0">A classic dataset for classification, pattern recognition and machine learning.</p>
                </div>
                <div className="grid md:grid-cols-2 gap-6 mb-6">
                    <div className="bg-muted/30 p-4 rounded-lg">
                        <h5 className="font-medium mb-2">Dataset Size</h5>
                        <p className="mb-0">150 samples of iris flowers</p>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg">
                        <h5 className="font-medium mb-2">Target Classes</h5>
                        <p className="mb-0">3 species: Setosa, Versicolor, Virginica</p>
                    </div>
                </div>
                <h4 className="text-lg font-semibold">Features</h4>
                <p>The dataset includes four numerical features, all measured in centimeters:</p>
                <div className="grid grid-cols-2 gap-4 my-4">
                    <div className="border rounded-md p-3">
                        <div className="font-medium">Sepal Length</div>
                    </div>
                    <div className="border rounded-md p-3">
                        <div className="font-medium">Sepal Width</div>
                    </div>
                    <div className="border rounded-md p-3">
                        <div className="font-medium">Petal Length</div>
                    </div>
                    <div className="border rounded-md p-3">
                        <div className="font-medium">Petal Width</div>
                    </div>
                </div>
                <p>
                    Additionally, the <strong>species</strong> column represents the target variable, which helps distinguish between different flower types. This dataset is particularly useful for understanding how numerical features vary across categories and identifying potential trends.
                </p>
            </>
        )
    }
    if (section === 2) {
        return (
        <>
          <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
            <h4 className="mt-0 text-lg font-semibold">EDA Process</h4>
            <p className="mb-0">A step-by-step approach to understanding your data before modeling.</p>
          </div>

          <div className="space-y-8">
            <div>
              <h4 className="flex items-center gap-2 text-lg font-semibold">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-sm">
                  1
                </span>
                Importing Required Libraries
              </h4>
              <p>To perform EDA, we need libraries for data manipulation and visualization:</p>
              <ul className="space-y-1 list-disc pl-6 mb-4">
                <li>
                  <strong>pandas</strong> - for handling tabular data
                </li>
                <li>
                  <strong>seaborn</strong> - for statistical visualizations
                </li>
                <li>
                  <strong>matplotlib</strong> - for general plotting
                </li>
              </ul>

              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() =>
                      copyToClipboard(
                        `import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline   
sns.set(style='darkgrid')`,
                        "code1",
                      )
                    }
                  >
                    {copied === "code1" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto">
                  <code>
                    {`import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline   
sns.set(style='darkgrid')`}
                  </code>
                </pre>
              </div>
              <p className="text-sm text-muted-foreground">
                The <code>%matplotlib inline</code> command ensures plots display directly in Jupyter
                Notebook.
              </p>
            </div>

            <div>
              <h4 className="flex items-center gap-2 text-lg font-semibold">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-sm">
                  2
                </span>
                Loading the Dataset
              </h4>
              <p>
                Next, we load the Iris Dataset using seaborn's built-in function and display the first five
                rows:
              </p>

              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() =>
                      copyToClipboard(
                        `df = sns.load_dataset('iris') 
df.head()  # Display first five rows`,
                        "code2",
                      )
                    }
                  >
                    {copied === "code2" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto">
                  <code>
                    {`df = sns.load_dataset('iris') 
df.head()  # Display first five rows`}
                  </code>
                </pre>
              </div>
              <p className="text-sm text-muted-foreground">
                This helps verify the dataset is correctly loaded and gives an overview of column names and
                sample values.
              </p>
            </div>

            <div>
              <h4 className="flex items-center gap-2 text-lg font-semibold">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-sm">
                  3
                </span>
                Checking Data Types
              </h4>
              <p>Understanding the data types of each column is essential for further analysis:</p>

              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => copyToClipboard(`df.dtypes`, "code3")}
                  >
                    {copied === "code3" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto">
                  <code>{`df.dtypes`}</code>
                </pre>
              </div>
              <p className="text-sm text-muted-foreground">
                Ensuring correct data types is crucial to avoid errors when performing numerical operations
                or visualizations.
              </p>
            </div>

            <div>
              <h4 className="flex items-center gap-2 text-lg font-semibold">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-sm">
                  4
                </span>
                Checking for Missing Values
              </h4>
              <p>Missing values can significantly impact analysis, leading to biased results or errors:</p>

              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => copyToClipboard(`df.isnull().sum()`, "code4")}
                  >
                    {copied === "code4" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto">
                  <code>{`df.isnull().sum()`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h4 className="flex items-center gap-2 text-lg font-semibold">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground text-sm">
                  5
                </span>
                Summary Statistics
              </h4>
              <p>To gain insights into the distribution of numerical features:</p>

              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() => copyToClipboard(`df.describe()`, "code5")}
                  >
                    {copied === "code5" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto">
                  <code>{`df.describe()`}</code>
                </pre>
              </div>
              <p className="text-sm text-muted-foreground">
                This provides key statistical metrics such as mean, standard deviation, minimum and maximum
                values, and quartiles.
              </p>
            </div>
          </div>
        </>
      )}
      if (section === 3) {
        return (
        <>
          <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
            <h4 className="mt-0 text-lg font-semibold">Data Visualization</h4>
            <p className="mb-0">
              Visualizing data helps in understanding relationships and variations between features.
            </p>
          </div>

          <div className="space-y-8">
            <div>
              <h4 className="text-lg font-semibold">Pairplot Visualization</h4>
              <p>
                A pairplot shows scatter plots of feature pairs and highlights differences among species:
              </p>

              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() =>
                      copyToClipboard(
                        `sns.pairplot(df, hue='species') 
plt.show()`,
                        "code6",
                      )
                    }
                  >
                    {copied === "code6" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto">
                  <code>
                    {`sns.pairplot(df, hue='species') 
plt.show()`}
                  </code>
                </pre>
              </div>

              <div className="bg-muted/30 p-4 rounded-lg mt-4">
                <h5 className="font-medium mb-2">What to Look For:</h5>
                <ul className="space-y-1 list-disc pl-6 mb-0">
                  <li>Clear separation between species clusters</li>
                  <li>Relationships between different features</li>
                  <li>Potential outliers in the data</li>
                </ul>
              </div>
            </div>

            <div>
              <h4 className="text-lg font-semibold">Boxplot Visualization</h4>
              <p>Boxplots help examine feature distributions and detect potential outliers:</p>

              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() =>
                      copyToClipboard(
                        `sns.boxplot(data=df.drop(columns=['species'])) 
plt.show()`,
                        "code7",
                      )
                    }
                  >
                    {copied === "code7" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto">
                  <code>
                    {`sns.boxplot(data=df.drop(columns=['species'])) 
plt.show()`}
                  </code>
                </pre>
              </div>

              <div className="bg-muted/30 p-4 rounded-lg mt-4">
                <h5 className="font-medium mb-2">Understanding Boxplots:</h5>
                <ul className="space-y-1 list-disc pl-6 mb-0">
                  <li>The box represents the interquartile range (IQR)</li>
                  <li>The line inside the box is the median</li>
                  <li>Whiskers extend to 1.5 * IQR</li>
                  <li>Points beyond whiskers are potential outliers</li>
                </ul>
              </div>
            </div>
          </div>
        </>
      )}
    if (section === 4) {
        return (
        <>
          <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
            <h4 className="mt-0 text-lg font-semibold">Understanding Feature Relationships</h4>
            <p className="mb-0">
              Analyzing correlations between numerical features helps identify dependencies.
            </p>
          </div>

          <div>
            <h4 className="text-lg font-semibold">Correlation Heatmap</h4>
            <p>A heatmap visualizes correlation coefficients between variables:</p>

            <div className="relative bg-black rounded-md my-4 group">
              <div className="absolute right-2 top-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={() =>
                    copyToClipboard(
                      `sns.heatmap(df.corr(), annot=True, cmap='coolwarm') 
plt.show()`,
                      "code8",
                    )
                  }
                >
                  {copied === "code8" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                </Button>
              </div>
              <pre className="p-4 text-white overflow-x-auto">
                <code>
                  {`sns.heatmap(df.corr(), annot=True, cmap='coolwarm') 
plt.show()`}
                </code>
              </pre>
            </div>

            <div className="grid md:grid-cols-3 gap-4 my-6">
              <div className="bg-green-50 dark:bg-green-950 p-4 rounded-lg border border-green-200 dark:border-green-800">
                <h5 className="font-medium mb-2 text-green-700 dark:text-green-400">
                  Positive Correlation (1)
                </h5>
                <p className="text-sm mb-0">Both variables increase together</p>
              </div>
              <div className="bg-red-50 dark:bg-red-950 p-4 rounded-lg border border-red-200 dark:border-red-800">
                <h5 className="font-medium mb-2 text-red-700 dark:text-red-400">
                  Negative Correlation (-1)
                </h5>
                <p className="text-sm mb-0">One increases, the other decreases</p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-200 dark:border-gray-800">
                <h5 className="font-medium mb-2 text-gray-700 dark:text-gray-400">No Correlation (0)</h5>
                <p className="text-sm mb-0">No relationship between variables</p>
              </div>
            </div>

            <div className="bg-muted/30 p-4 rounded-lg mt-4">
              <h5 className="font-medium mb-2">Key Insight:</h5>
              <p className="mb-0">
                In the Iris Dataset, <strong>Petal Length</strong> and <strong>Petal Width</strong> show a
                high correlation, meaning that one of them could potentially be redundant when building
                models.
              </p>
            </div>

            <div className="mt-6">
              <h4 className="text-lg font-semibold">Additional Correlation Analysis</h4>
              <p>For a more detailed analysis, you can also examine correlations within each species:</p>

              <div className="relative bg-black rounded-md my-4 group">
                <div className="absolute right-2 top-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-white"
                    onClick={() =>
                      copyToClipboard(
                        `# For each species
for species in df['species'].unique():
print(f"Correlation for {species}:")
print(df[df['species'] == species].drop('species', axis=1).corr())
print("\\n")`,
                        "code9",
                      )
                    }
                  >
                    {copied === "code9" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
                <pre className="p-4 text-white overflow-x-auto">
                  <code>
                    {`# For each species
for species in df['species'].unique():
print(f"Correlation for {species}:")
print(df[df['species'] == species].drop('species', axis=1).corr())
print("\\n")`}
                  </code>
                </pre>
              </div>
            </div>
          </div>
        </>
      )}
      if (section === 5) {
        return (
        <>
          <div className="bg-muted/50 p-4 rounded-lg mb-6 border-l-4 border-primary">
            <h4 className="mt-0 text-lg font-semibold">Key Takeaways</h4>
            <p className="mb-0">
              EDA is a crucial step in data analysis that allows us to explore datasets before applying
              machine learning models.
            </p>
          </div>

          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-muted/30 p-4 rounded-lg border border-muted">
                <h5 className="font-medium mb-3">What We Learned</h5>
                <ul className="space-y-2 list-disc pl-6 mb-0">
                  <li>How to load and inspect dataset structure</li>
                  <li>Techniques for checking data quality</li>
                  <li>Methods to visualize feature distributions</li>
                  <li>Ways to identify relationships between variables</li>
                </ul>
              </div>
              <div className="bg-muted/30 p-4 rounded-lg border border-muted">
                <h5 className="font-medium mb-3">Why It Matters</h5>
                <ul className="space-y-2 list-disc pl-6 mb-0">
                  <li>Prevents building models on flawed data</li>
                  <li>Helps select relevant features</li>
                  <li>Guides feature engineering decisions</li>
                  <li>Improves model performance and reliability</li>
                </ul>
              </div>
            </div>

            <div className="bg-primary/10 p-6 rounded-lg border border-primary/20">
              <h4 className="text-lg font-semibold mb-4">EDA Process Summary</h4>
              <ol className="space-y-3 list-decimal pl-6">
                <li>
                  <strong>Data Loading & Inspection</strong>
                  <p className="text-sm mt-1 mb-0">Loaded the dataset and examined its structure</p>
                </li>
                <li>
                  <strong>Data Cleaning & Validation</strong>
                  <p className="text-sm mt-1 mb-0">Checked for missing values and data types</p>
                </li>
                <li>
                  <strong>Statistical Analysis</strong>
                  <p className="text-sm mt-1 mb-0">
                    Generated summary statistics to understand distributions
                  </p>
                </li>
                <li>
                  <strong>Data Visualization</strong>
                  <p className="text-sm mt-1 mb-0">
                    Created visualizations to identify patterns and relationships
                  </p>
                </li>
                <li>
                  <strong>Correlation Analysis</strong>
                  <p className="text-sm mt-1 mb-0">Examined relationships between features</p>
                </li>
              </ol>
            </div>

            <div className="mt-8 text-center">
              <h4 className="text-lg font-semibold mb-4">Next Steps</h4>
              <p>
                Now that you understand the basics of EDA, you're ready to apply these techniques to your
                own datasets.
              </p>
              <div className="flex justify-center gap-4 mt-6">
                <Button variant="outline">Download Notebook</Button>
                <Button>Try Advanced EDA</Button>
              </div>
            </div>
          </div>
        </>
      )}
    return null
  }
  

function copyToClipboard(arg0: string, arg1: string): void {
    throw new Error("Function not implemented.")
}
  