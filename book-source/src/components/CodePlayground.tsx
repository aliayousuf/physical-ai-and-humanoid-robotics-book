import React, { useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

const CodePlayground = ({ initialCode, language = 'python', title = 'Interactive Code Playground' }) => {
  const [code, setCode] = useState(initialCode);

  const executeCode = () => {
    // In a real implementation, this would connect to a backend service
    // to safely execute the code. For now, we'll just show an alert.
    if (ExecutionEnvironment.canUseDOM) {
      alert('In a real implementation, this would execute the code in a safe sandboxed environment.');
    }
  };

  return (
    <div className="code-playground-container">
      <div className="playground-header">
        <h4>{title}</h4>
      </div>
      <div className="playground-controls">
        <button onClick={executeCode} className="run-button">
          Run Code
        </button>
        <button
          onClick={() => setCode(initialCode)}
          className="reset-button"
        >
          Reset
        </button>
      </div>
      <div className="playground-editor">
        <BrowserOnly>
          {() => {
            const CodeEditor = require('@uiw/react-textarea-code-editor').default;
            return (
              <CodeEditor
                value={code}
                language={language}
                onChange={(evn) => setCode(evn.target.value)}
                padding={15}
                style={{
                  fontSize: 12,
                  backgroundColor: '#f5f5f5',
                  fontFamily: 'ui-monospace,SFMono-Regular,"SF Mono",Consolas,"Liberation Mono",Menlo,monospace',
                }}
              />
            );
          }}
        </BrowserOnly>
      </div>
      <div className="playground-output">
        <h5>Output:</h5>
        <pre className="output-area">
          {ExecutionEnvironment.canUseDOM
            ? 'Output will appear here after running the code'
            : 'Run the code to see output'}
        </pre>
      </div>
    </div>
  );
};

export default CodePlayground;