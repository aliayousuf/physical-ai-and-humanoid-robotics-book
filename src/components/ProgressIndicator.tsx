import React, { useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import { useProgressTracker } from './ProgressTracker';

interface ProgressIndicatorProps {
  chapterId: string;
  title: string;
}

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ chapterId, title }) => {
  const location = useLocation();
  const { markComplete, isComplete } = useProgressTracker();
  const [completed, setCompleted] = React.useState(false);

  useEffect(() => {
    setCompleted(isComplete(chapterId));
  }, [chapterId]);

  // Mark as complete when user has spent some time on the page
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!completed) {
        markComplete(chapterId);
        setCompleted(true);
      }
    }, 30000); // Mark as complete after 30 seconds on page

    return () => clearTimeout(timer);
  }, [chapterId, completed, markComplete]);

  return (
    <div className="progress-indicator" style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'flex-end',
      marginBottom: '1rem',
      padding: '0.5rem',
      border: '1px solid #e0e0e0',
      borderRadius: '4px',
      backgroundColor: '#f9f9f9'
    }}>
      <span style={{
        marginRight: '0.5rem',
        fontSize: '0.9rem',
        color: '#666'
      }}>
        {completed ? '✓ Completed' : 'Track Progress'}
      </span>
      <div style={{
        width: '20px',
        height: '20px',
        borderRadius: '50%',
        backgroundColor: completed ? '#4caf50' : '#e0e0e0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.7rem',
        color: completed ? 'white' : '#666'
      }}>
        {completed ? '✓' : '○'}
      </div>
    </div>
  );
};

export default ProgressIndicator;