import React from 'react';

const Troubleshooting = ({ title = "Troubleshooting", issues }) => {
  return (
    <div className="troubleshooting-container">
      <h3>{title}</h3>
      <div className="troubleshooting-list">
        {issues.map((issue, index) => (
          <div key={index} className="troubleshooting-item">
            <h4>Issue: {issue.problem}</h4>
            <p><strong>Cause:</strong> {issue.cause}</p>
            <p><strong>Solution:</strong> {issue.solution}</p>
            {issue.command && (
              <div className="solution-command">
                <code>{issue.command}</code>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Troubleshooting;