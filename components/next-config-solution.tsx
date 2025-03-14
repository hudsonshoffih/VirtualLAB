import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Code } from "@/components/ui/code"

export default function NextConfigSolution() {
  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Next.js ESLint Configuration Solutions</CardTitle>
        <CardDescription>Choose the best approach for your project</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="quick-fix">
          <TabsList className="grid grid-cols-2 mb-4">
            <TabsTrigger value="quick-fix">Quick Fix</TabsTrigger>
            <TabsTrigger value="proper-fix">Proper Fix</TabsTrigger>
          </TabsList>

          <TabsContent value="quick-fix">
            <div className="space-y-4">
              <p>
                For a quick fix to get your deployment working, create or modify your <code>.eslintrc.js</code> file in
                the root of your project with the following content:
              </p>

              <Code className="language-javascript">
                {`module.exports = {
  extends: 'next/core-web-vitals',
  rules: {
    // Disable the rules that are causing the build to fail
    'react/no-unescaped-entities': 'off',
    '@typescript-eslint/no-explicit-any': 'off',
    '@typescript-eslint/no-unused-vars': 'off',
    '@next/next/no-img-element': 'off'
  }
}`}
              </Code>

              <p>
                This will disable the ESLint rules that are causing your build to fail, allowing your deployment to
                complete. However, this is only a temporary solution and you should address these issues properly when
                you have time.
              </p>
            </div>
          </TabsContent>

          <TabsContent value="proper-fix">
            <div className="space-y-4">
              <p>For a proper fix, you should address each type of issue:</p>

              <h3 className="text-lg font-medium">1. Unescaped Entities</h3>
              <p>Replace apostrophes and quotes in JSX with their HTML entity equivalents:</p>
              <Code className="language-jsx">
                {`// Before
<p>Don't use apostrophes directly</p>

// After
<p>Don{"'"}t use apostrophes directly</p>
// or
<p>Don&apos;t use apostrophes directly</p>`}
              </Code>

              <h3 className="text-lg font-medium">2. TypeScript 'any' Types</h3>
              <p>Replace 'any' with proper type definitions:</p>
              <Code className="language-typescript">
                {`// Before
function processData(data: any) {
  // ...
}

// After
interface DataType {
  id: string;
  name: string;
  // other properties...
}

function processData(data: DataType) {
  // ...
}`}
              </Code>

              <h3 className="text-lg font-medium">3. Unused Variables</h3>
              <p>Either remove unused variables or prefix them with an underscore:</p>
              <Code className="language-typescript">
                {`// Before
function Component({ data, unused }) {
  // 'unused' is never used
  return <div>{data}</div>;
}

// After - Option 1: Remove it
function Component({ data }) {
  return <div>{data}</div>;
}

// After - Option 2: Prefix with underscore
function Component({ data, _unused }) {
  return <div>{data}</div>;
}`}
              </Code>

              <h3 className="text-lg font-medium">4. Image Elements</h3>
              <p>Replace HTML img tags with Next.js Image component:</p>
              <Code className="language-jsx">
                {`// Before
<img src="/image.jpg" alt="Description" />

// After
import Image from 'next/image';

<Image 
  src="/image.jpg" 
  alt="Description" 
  width={500} 
  height={300} 
/>`}
              </Code>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

