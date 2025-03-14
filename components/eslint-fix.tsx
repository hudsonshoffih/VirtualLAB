"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { useState } from "react"

export default function ESLintFixer() {
  const [output, setOutput] = useState("")

  const fixUnescapedEntities = () => {
    const result = `
// Create a .eslintrc.js file in your project root with this content:

module.exports = {
  extends: 'next/core-web-vitals',
  rules: {
    // Disable the rule for unescaped entities
    'react/no-unescaped-entities': 'off',
    
    // Optionally disable the any type rule if you need time to fix them properly
    '@typescript-eslint/no-explicit-any': 'warn',
    
    // Optionally disable unused vars rule if you need time to fix them
    '@typescript-eslint/no-unused-vars': 'warn',
    
    // Optionally disable the img element warning
    '@next/next/no-img-element': 'warn'
  }
}
`
    setOutput(result)
  }

  const fixAnyTypes = () => {
    const result = `
// To fix 'any' types, you should define proper interfaces or types.
// Here's an example of how to replace 'any' with proper types:

// Before:
const handleData = (data: any) => {
  console.log(data)
}

// After:
interface DataType {
  id: string;
  name: string;
  // add other properties as needed
}

const handleData = (data: DataType) => {
  console.log(data)
}

// For arrays:
// Before: 
const items: any[] = [];

// After:
const items: DataType[] = [];

// If you're not sure about the type yet, use 'unknown' instead of 'any':
const handleUnknownData = (data: unknown) => {
  // You'll need to check the type before using it
  if (typeof data === 'object' && data !== null) {
    // Now you can use it, but still with caution
  }
}
`
    setOutput(result)
  }

  const fixUnusedVars = () => {
    const result = `
// To fix unused variables, you have several options:

// 1. Remove the unused variable
// Before:
const { data, unused } = props;
// After:
const { data } = props;

// 2. Prefix with underscore to indicate it's intentionally unused
// Before:
const { data, unused } = props;
// After:
const { data, _unused } = props;

// 3. Use destructuring to skip variables
// Before:
const [value, setValue, unused] = useState();
// After:
const [value, setValue] = useState();

// 4. For function parameters you don't use:
// Before:
function example(param1, param2) {
  console.log(param1);
  // param2 is unused
}
// After:
function example(param1, _param2) {
  console.log(param1);
}
`
    setOutput(result)
  }

  const fixImgElements = () => {
    const result = `
// Replace <img> elements with Next.js <Image> component:

// Before:
<img src="/image.jpg" alt="Description" />

// After:
import Image from 'next/image';

<Image 
  src="/image.jpg" 
  alt="Description" 
  width={500} 
  height={300} 
  // Optional: make it responsive
  layout="responsive"
/>

// For images with unknown dimensions, you can use:
<Image 
  src="/image.jpg" 
  alt="Description" 
  fill
  style={{ objectFit: 'cover' }}
/>
`
    setOutput(result)
  }

  const createFixScript = () => {
    const result = `
// Create a file called fix-eslint-errors.js in your project root:

const fs = require('fs');
const path = require('path');

// Function to recursively find all .tsx and .ts files
function findFiles(dir, fileList = []) {
  const files = fs.readdirSync(dir);
  
  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory() && !filePath.includes('node_modules') && !filePath.includes('.next')) {
      fileList = findFiles(filePath, fileList);
    } else if (
      stat.isFile() && 
      (filePath.endsWith('.tsx') || filePath.endsWith('.ts')) && 
      !filePath.includes('.d.ts')
    ) {
      fileList.push(filePath);
    }
  });
  
  return fileList;
}

// Function to fix unescaped entities
function fixUnescapedEntities(content) {
  // Replace ' with &apos;
  content = content.replace(/(\${|{|\`|"|>)([^<>]*?)'([^<>]*?)(<|\`|"|})(?![^<>]*?<\/code>)/g, '$1$2&apos;$3$4');
  
  // Replace " with &quot;
  content = content.replace(/(\${|{|\`|'|>)([^<>]*?)"([^<>]*?)(<|\`|'|})(?![^<>]*?<\/code>)/g, '$1$2&quot;$3$4');
  
  return content;
}

// Function to process a file
function processFile(filePath) {
  console.log(\`Processing \${filePath}\`);
  
  let content = fs.readFileSync(filePath, 'utf8');
  const originalContent = content;
  
  // Fix unescaped entities
  content = fixUnescapedEntities(content);
  
  // Only write if content changed
  if (content !== originalContent) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(\`Fixed unescaped entities in \${filePath}\`);
  }
}

// Main function
function main() {
  const rootDir = process.cwd();
  const files = findFiles(rootDir);
  
  console.log(\`Found \${files.length} files to process\`);
  
  files.forEach(file => {
    processFile(file);
  });
  
  console.log('Done!');
}

main();
`
    setOutput(result)
  }

  const createEslintConfig = () => {
    const result = `
// .eslintrc.js
module.exports = {
  extends: 'next/core-web-vitals',
  rules: {
    // Turn off rules that are causing the build to fail
    'react/no-unescaped-entities': 'off',
    '@typescript-eslint/no-explicit-any': 'off',
    '@typescript-eslint/no-unused-vars': 'off',
    '@next/next/no-img-element': 'off'
  }
}
`
    setOutput(result)
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>ESLint Error Fixer</CardTitle>
        <CardDescription>Tools to fix ESLint errors in your Next.js project</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="quick-fix">
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="quick-fix">Quick Fix</TabsTrigger>
            <TabsTrigger value="detailed-fixes">Detailed Fixes</TabsTrigger>
            <TabsTrigger value="script">Fix Script</TabsTrigger>
          </TabsList>

          <TabsContent value="quick-fix">
            <p className="mb-4">
              The quickest way to fix your deployment is to disable the ESLint rules that are causing the build to fail:
            </p>
            <Button onClick={createEslintConfig} className="mb-4">
              Generate ESLint Config
            </Button>
            <p className="mb-4">
              This is a temporary solution. For a proper fix, you should address each issue individually.
            </p>
          </TabsContent>

          <TabsContent value="detailed-fixes">
            <div className="grid grid-cols-2 gap-4 mb-4">
              <Button onClick={fixUnescapedEntities}>Fix Unescaped Entities</Button>
              <Button onClick={fixAnyTypes}>Fix 'any' Types</Button>
              <Button onClick={fixUnusedVars}>Fix Unused Variables</Button>
              <Button onClick={fixImgElements}>Fix img Elements</Button>
            </div>
          </TabsContent>

          <TabsContent value="script">
            <p className="mb-4">Generate a script to automatically fix unescaped entities in your project:</p>
            <Button onClick={createFixScript} className="mb-4">
              Generate Fix Script
            </Button>
            <p className="mb-4">
              Run this script with: <code>node fix-eslint-errors.js</code>
            </p>
          </TabsContent>

          {output && (
            <div className="mt-4">
              <h3 className="text-lg font-medium mb-2">Solution:</h3>
              <Textarea value={output} readOnly className="font-mono text-sm h-80" />
            </div>
          )}
        </Tabs>
      </CardContent>
    </Card>
  )
}

