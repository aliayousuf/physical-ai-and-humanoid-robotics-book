import React, { useState, useEffect } from 'react';
import { validateProgressData, validateChapterId } from '../utils/validation';

interface ProgressItem {
  id: string;
  completed: boolean;
  timestamp: number;
}

interface ProgressData {
  [key: string]: ProgressItem;
}

const ProgressTracker = () => {
  const [progress, setProgress] = useState<ProgressData>({});
  const [loading, setLoading] = useState(true);

  // Load progress from localStorage on component mount
  useEffect(() => {
    const savedProgress = localStorage.getItem('robotics-book-progress');
    if (savedProgress) {
      try {
        const parsedProgress = JSON.parse(savedProgress);
        // Validate the loaded progress data
        const validProgress: ProgressData = {};
        for (const [key, value] of Object.entries(parsedProgress)) {
          if (validateProgressData(value)) {
            validProgress[key] = value as ProgressItem;
          }
        }
        setProgress(validProgress);
      } catch (error) {
        console.error('Error parsing progress data from localStorage:', error);
        setProgress({});
      }
    }
    setLoading(false);
  }, []);

  // Save progress to localStorage whenever progress changes
  useEffect(() => {
    if (!loading) {
      try {
        localStorage.setItem('robotics-book-progress', JSON.stringify(progress));
      } catch (error) {
        console.error('Error saving progress data to localStorage:', error);
      }
    }
  }, [progress, loading]);

  const updateProgress = (id: string, completed: boolean) => {
    if (!validateChapterId(id)) {
      console.error('Invalid chapter ID provided to updateProgress');
      return;
    }

    setProgress(prev => ({
      ...prev,
      [id]: {
        id,
        completed,
        timestamp: Date.now()
      }
    }));
  };

  const getProgress = (id: string): ProgressItem | undefined => {
    return progress[id];
  };

  const getOverallProgress = (): number => {
    const allItems = Object.values(progress);
    if (allItems.length === 0) return 0;
    const completedItems = allItems.filter(item => item.completed).length;
    return Math.round((completedItems / allItems.length) * 100);
  };

  return (
    <div className="progress-tracker" style={{ display: 'none' }}>
      {/* This component is used for progress tracking functionality */}
      {/* It's hidden since progress tracking happens in the background */}
    </div>
  );
};

// Export utility functions for use in other components
export const useProgressTracker = () => {
  const markComplete = (id: string) => {
    if (!validateChapterId(id)) {
      console.error('Invalid chapter ID provided to markComplete');
      return;
    }

    try {
      const savedProgress = localStorage.getItem('robotics-book-progress');
      let progress: ProgressData = {};

      if (savedProgress) {
        const parsedProgress = JSON.parse(savedProgress);
        // Validate existing progress data
        for (const [key, value] of Object.entries(parsedProgress)) {
          if (validateProgressData(value)) {
            progress[key] = value as ProgressItem;
          }
        }
      }

      progress[id] = {
        id,
        completed: true,
        timestamp: Date.now()
      };

      localStorage.setItem('robotics-book-progress', JSON.stringify(progress));
    } catch (error) {
      console.error('Error updating progress:', error);
    }
  };

  const isComplete = (id: string): boolean => {
    if (!validateChapterId(id)) {
      console.error('Invalid chapter ID provided to isComplete');
      return false;
    }

    try {
      const savedProgress = localStorage.getItem('robotics-book-progress');
      if (!savedProgress) return false;

      const progress: ProgressData = JSON.parse(savedProgress);
      // Validate the progress item before returning
      const item = progress[id];
      if (item && validateProgressData(item)) {
        return item.completed || false;
      }
      return false;
    } catch (error) {
      console.error('Error checking progress:', error);
      return false;
    }
  };

  const getProgressPercentage = (): number => {
    try {
      const savedProgress = localStorage.getItem('robotics-book-progress');
      if (!savedProgress) return 0;

      const progress: ProgressData = JSON.parse(savedProgress);
      // Filter only valid progress items
      const validItems = Object.values(progress).filter(item => validateProgressData(item));
      if (validItems.length === 0) return 0;

      const completedItems = validItems.filter(item => item.completed).length;
      return Math.round((completedItems / validItems.length) * 100);
    } catch (error) {
      console.error('Error calculating progress percentage:', error);
      return 0;
    }
  };

  return { markComplete, isComplete, getProgressPercentage };
};

export default ProgressTracker;