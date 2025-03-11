"use client"

import React from "react"
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  TooltipProps,
} from "recharts"

interface ChartProps {
  type: "scatter" | "line" | "histogram" | "heatmap"
  data: any
}

export function Chart({ type, data }: ChartProps) {
  if (type === "scatter") {
    const chartData = data.x.map((x: number, i: number) => ({
      x,
      y: data.y[i],
      group: data.groups ? data.groups[i] : "default",
    }))

    const groups = Array.from(new Set(data.groups || ["default"]))
    const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#0088fe"]

    return (
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid />
          <XAxis type="number" dataKey="x" name="X" />
          <YAxis type="number" dataKey="y" name="Y" />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Legend />
          {groups.map((group, index) => (
            <Scatter
              key={group as string}
              name={group as string}
              data={chartData.filter((d: any) => d.group === group)}
              fill={colors[index % colors.length]}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    )
  }

  if (type === "line") {
    const chartData = data.x.map((x: number, i: number) => ({
      x,
      actual: data.y[i],
      predicted: data.predicted ? data.predicted[i] : null,
    }))

    return (
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" />
          {data.predicted && (
            <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" strokeDasharray="5 5" />
          )}
        </LineChart>
      </ResponsiveContainer>
    )
  }

  if (type === "histogram") {
    // Convert histogram data to bar chart format
    const histogramData: { bin: string; count: number }[] = []
    const min = Math.min(...data.values)
    const max = Math.max(...data.values)
    const binWidth = (max - min) / data.bins

    // Create bins
    const bins = Array.from({ length: data.bins }, (_, i) => ({
      binStart: min + i * binWidth,
      binEnd: min + (i + 1) * binWidth,
      count: 0,
    }))

    // Count values in each bin
    data.values.forEach((value: number) => {
      const binIndex = Math.min(Math.floor((value - min) / binWidth), data.bins - 1)
      bins[binIndex].count++
    })

    // Format for chart
    bins.forEach((bin) => {
      histogramData.push({
        bin: `${bin.binStart.toFixed(1)}-${bin.binEnd.toFixed(1)}`,
        count: bin.count,
      })
    })

    return (
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={histogramData} margin={{ top: 20, right: 20, bottom: 60, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="bin" angle={-45} textAnchor="end" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="count" fill="#8884d8" name="Frequency" />
        </BarChart>
      </ResponsiveContainer>
    )
  }

  if (type === "heatmap") {
    // For confusion matrix
    return (
      <div className="flex flex-col items-center">
        <h3 className="text-lg font-medium mb-4">Confusion Matrix</h3>
        <div
          className="grid"
          style={{
            gridTemplateColumns: `repeat(${data.matrix[0].length + 1}, minmax(60px, 1fr))`,
            gap: "1px",
          }}
        >
          {/* Empty top-left cell */}
          <div className="bg-muted p-2 text-center font-medium"></div>

          {/* Column headers */}
          {data.labels.map((label: string, i: number) => (
            <div key={`col-${i}`} className="bg-muted p-2 text-center font-medium">
              {label}
            </div>
          ))}

          {/* Row headers and data */}
          {data.matrix.map((row: number[], rowIndex: number) => (
            <React.Fragment key={`row-${rowIndex}`}>
              {/* Row header */}
              <div className="bg-muted p-2 text-center font-medium">{data.labels[rowIndex]}</div>

              {/* Row data */}
              {row.map((cell: number, colIndex: number) => {
                // Calculate color intensity based on value
                const maxValue = Math.max(...data.matrix.flat())
                const intensity = cell / maxValue
                const bgColor = `rgba(136, 132, 216, ${intensity})`

                return (
                  <div
                    key={`cell-${rowIndex}-${colIndex}`}
                    className="p-4 text-center font-bold border"
                    style={{ backgroundColor: bgColor }}
                  >
                    {cell}
                  </div>
                )
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
    )
  }

  return <div>Unsupported chart type</div>
}

// Chart container component
export const ChartContainer = ({ children }: { children: React.ReactNode }) => (
  <div className="w-full h-[400px] my-4">{children}</div>
)

// Chart title component
export const ChartTitle = ({ children }: { children: React.ReactNode }) => (
  <h3 className="text-lg font-medium mb-4 text-center">{children}</h3>
)

// Chart tooltip components - Updated to support render props
type TooltipRenderProps = {
  point: {
    data: any;
    [key: string]: any;
  };
  [key: string]: any;
};

export const ChartTooltip = ({ children }: { children: ((props: TooltipRenderProps) => React.ReactNode) | React.ReactNode }) => {
  // This is now a wrapper that can accept either direct children or a render prop function
  return <>{children}</>
}

export const ChartTooltipContent = ({ children }: { children: React.ReactNode }) => (
  <div className="bg-background border rounded-md shadow-md p-2">{children}</div>
)

export const ChartTooltipItem = ({ label, value }: { label: string; value: string | number }) => (
  <div className="flex justify-between gap-2">
    <span className="font-medium">{label}:</span>
    <span>{value}</span>
  </div>
)

// Chart legend components
export const ChartLegend = ({ children }: { children: React.ReactNode }) => (
  <div className="chart-legend mt-4 flex flex-wrap justify-center gap-4">{children}</div>
)

export const ChartLegendItem = ({ color, label }: { color: string; label: string }) => (
  <div className="flex items-center gap-2">
    <div className="w-3 h-3" style={{ backgroundColor: color }}></div>
    <span>{label}</span>
  </div>
)

// Bar chart components
export const ChartBar = ({ 
  data, 
  children 
}: { 
  data: any[]; 
  children: ((data: any) => React.ReactNode) | React.ReactNode 
}) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="category" />
        <YAxis />
        <Tooltip content={(props) => {
          if (typeof children === 'function') {
            // If children is a function, we can use it to render tooltip content
            const point = { data: props.payload?.[0]?.payload || {} };
            return <div className="custom-tooltip bg-white p-2 border rounded shadow">
              {children({ point })}
            </div>;
          }
          return null;
        }} />
        <Legend />
        {typeof children !== 'function' ? children : children(data)}
      </BarChart>
    </ResponsiveContainer>
  )
}

export const ChartBarItem = ({ 
  data, 
  valueAccessor, 
  categoryAccessor, 
  style 
}: { 
  data: any[]; 
  valueAccessor: (d: any) => number; 
  categoryAccessor: (d: any) => string;
  style?: any;
}) => {
  return (
    <Bar dataKey={(d) => valueAccessor(d)} name="Value">
      {Array.isArray(data) && data.map((entry, index) => (
        <Cell 
          key={`cell-${index}`} 
          fill={style?.fill ? (typeof style.fill === 'function' ? style.fill(entry) : style.fill) : '#8884d8'} 
        />
      ))}
    </Bar>
  )
}

// Line chart components
export const ChartLine = ({ 
  data, 
  children 
}: { 
  data: any[]; 
  children: ((data: any) => React.ReactNode) | React.ReactNode 
}) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="category" />
        <YAxis />
        <Tooltip content={(props) => {
          if (typeof children === 'function') {
            // If children is a function, we can use it to render tooltip content
            const point = { data: props.payload?.[0]?.payload || {} };
            return <div className="custom-tooltip bg-white p-2 border rounded shadow">
              {children({ point })}
            </div>;
          }
          return null;
        }} />
        <Legend />
        {typeof children !== 'function' ? children : children(data)}
      </LineChart>
    </ResponsiveContainer>
  )
}

export const ChartLineItem = ({ 
  data, 
  valueAccessor, 
  categoryAccessor, 
  style 
}: { 
  data: any[]; 
  valueAccessor: (d: any) => number; 
  categoryAccessor: (d: any) => string;
  style?: any;
}) => {
  return (
    <Line 
      type="monotone" 
      dataKey={(d) => valueAccessor(d)} 
      stroke={style?.stroke || '#8884d8'} 
      strokeWidth={style?.strokeWidth || 2}
    />
  )
}

// Style component (placeholder)
export const ChartStyle = () => null