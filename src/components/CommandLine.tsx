import React from 'react';

const CommandLine = ({ command, output, explanation }) => {
  return (
    <div className="command-line-container">
      <div className="command-input">
        <span className="prompt">$</span> {command}
      </div>
      {output && (
        <div className="command-output">
          {output.split('\n').map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}
      {explanation && <div className="command-explanation">{explanation}</div>}
    </div>
  );
};

export default CommandLine;