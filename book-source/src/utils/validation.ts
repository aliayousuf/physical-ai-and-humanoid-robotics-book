/**
 * Input validation utilities for the Physical AI & Humanoid Robotics book
 */

// Validate progress tracking data
export const validateProgressData = (data: any): boolean => {
  if (!data || typeof data !== 'object') {
    console.error('Progress data must be an object');
    return false;
  }

  // Validate ID
  if (!data.id || typeof data.id !== 'string' || data.id.trim() === '') {
    console.error('Progress ID is required and must be a non-empty string');
    return false;
  }

  // Validate ID format (alphanumeric with hyphens/underscores)
  const idRegex = /^[a-zA-Z0-9_-]+$/;
  if (!idRegex.test(data.id)) {
    console.error('Progress ID must contain only alphanumeric characters, hyphens, or underscores');
    return false;
  }

  // Validate completed status
  if (typeof data.completed !== 'boolean') {
    console.error('Progress completed status must be a boolean');
    return false;
  }

  // Validate timestamp if present
  if (data.timestamp !== undefined) {
    if (typeof data.timestamp !== 'number' || data.timestamp <= 0) {
      console.error('Progress timestamp must be a positive number');
      return false;
    }
  }

  return true;
};

// Validate chapter ID format
export const validateChapterId = (id: string): boolean => {
  if (!id || typeof id !== 'string' || id.trim() === '') {
    console.error('Chapter ID is required and must be a non-empty string');
    return false;
  }

  // Check if ID follows expected format (module-number-chapter-name)
  const chapterIdRegex = /^module-[0-9]+-[a-z0-9-]+(?:\/[a-z0-9-]+)*$/;
  if (!chapterIdRegex.test(id)) {
    console.warn(`Chapter ID "${id}" does not follow expected format (e.g., "module-1-ros2/intro")`);
  }

  return true;
};

// Validate search query
export const validateSearchQuery = (query: string): boolean => {
  if (typeof query !== 'string') {
    console.error('Search query must be a string');
    return false;
  }

  if (query.length > 100) {
    console.error('Search query must be less than 100 characters');
    return false;
  }

  // Check for potentially harmful characters (basic XSS prevention)
  const dangerousChars = /[<>]/g;
  if (dangerousChars.test(query)) {
    console.error('Search query contains potentially dangerous characters');
    return false;
  }

  return true;
};

// Validate URL
export const validateUrl = (url: string): boolean => {
  if (typeof url !== 'string' || url.trim() === '') {
    console.error('URL must be a non-empty string');
    return false;
  }

  try {
    new URL(url);
    return true;
  } catch (e) {
    console.error(`Invalid URL: ${url}`);
    return false;
  }
};

// Validate numeric input
export const validateNumeric = (value: any, min?: number, max?: number): boolean => {
  if (typeof value !== 'number' || isNaN(value)) {
    console.error('Value must be a valid number');
    return false;
  }

  if (min !== undefined && value < min) {
    console.error(`Value must be greater than or equal to ${min}`);
    return false;
  }

  if (max !== undefined && value > max) {
    console.error(`Value must be less than or equal to ${max}`);
    return false;
  }

  return true;
};

// Validate string length
export const validateStringLength = (str: string, minLength: number, maxLength: number): boolean => {
  if (typeof str !== 'string') {
    console.error('Input must be a string');
    return false;
  }

  if (str.length < minLength) {
    console.error(`String must be at least ${minLength} characters long`);
    return false;
  }

  if (str.length > maxLength) {
    console.error(`String must be no more than ${maxLength} characters long`);
    return false;
  }

  return true;
};