import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import React from 'react';

type NotebookCellProps = {
  code: string;
  inputNumber: number;
};

const NotebookCell: React.FC<NotebookCellProps> = ({ code, inputNumber }) => {
  return (
    <div className="rounded-xl bg-gray-900 p-4 my-4 shadow-md border border-gray-700">
      <div className="text-green-400 font-mono mb-2">In [{inputNumber}]:</div>
      <SyntaxHighlighter language="python" style={oneDark} wrapLines={true}>
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

export default NotebookCell;
