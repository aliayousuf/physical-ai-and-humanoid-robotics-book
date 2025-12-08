# Data Model: Physical AI & Humanoid Robotics Book

## Book Module
**Description**: Educational content organized around a specific technology or concept area (ROS 2, Simulation, AI Pipelines, VLA)

**Fields**:
- `id` (string): Unique identifier for the module (e.g., "module-1-ros2")
- `title` (string): Display title of the module (e.g., "The Robotic Nervous System (ROS 2)")
- `description` (text): Overview of module content and learning objectives
- `order` (integer): Sequential position in the learning progression (1-4)
- `chapters` (array): References to Chapter entities belonging to this module
- `learningOutcomes` (array of strings): List of specific skills/knowledge learners will gain
- `prerequisites` (array of strings): List of required knowledge or completed modules
- `estimatedTime` (integer): Estimated hours to complete the module

**Relationships**:
- One-to-many with Chapter (one module contains many chapters)
- Many-to-one with Book (many modules belong to one book)

**Validation Rules**:
- `id` must be unique across all modules
- `order` must be between 1 and 4
- `chapters` array must contain at least one chapter
- `title` must not exceed 100 characters

## Chapter
**Description**: Individual sections within modules that progress from conceptual to technical to applied content

**Fields**:
- `id` (string): Unique identifier for the chapter (e.g., "module-1-chapter-1")
- `title` (string): Display title of the chapter
- `description` (text): Brief overview of chapter content
- `contentType` (enum): One of ["conceptual", "technical", "applied"]
- `module` (string): Reference to parent Book Module ID
- `order` (integer): Sequential position within the parent module
- `practicalExamples` (array): References to Practical Example entities
- `learningObjectives` (array of strings): List of specific learning objectives
- `duration` (integer): Estimated minutes to complete the chapter
- `prerequisites` (array of strings): List of required knowledge or completed chapters

**Relationships**:
- Many-to-one with Book Module (many chapters belong to one module)
- One-to-many with Practical Example (one chapter may contain many examples)

**Validation Rules**:
- `contentType` must be one of the allowed values
- `order` must be positive
- `module` must reference an existing Book Module

## Practical Example
**Description**: Hands-on exercises and code samples that allow users to implement concepts learned

**Fields**:
- `id` (string): Unique identifier for the example
- `title` (string): Display title of the example
- `description` (text): What the example demonstrates
- `chapter` (string): Reference to parent Chapter ID
- `codeFiles` (array of objects): List of code file objects with path and language
- `dependencies` (array of strings): List of required tools/technologies
- `expectedOutcome` (text): What user should achieve
- `difficulty` (enum): One of ["beginner", "intermediate", "advanced"]
- `estimatedTime` (integer): Estimated minutes to complete the example
- `downloadable` (boolean): Whether example files are available for download

**Code File Object Structure**:
- `path` (string): Relative path to the code file
- `language` (string): Programming language (e.g., "python", "cpp", "xml")
- `description` (text): Brief description of the file's purpose

**Relationships**:
- Many-to-one with Chapter (many examples belong to one chapter)

**Validation Rules**:
- `difficulty` must be one of the allowed values
- `codeFiles` array must not be empty if downloadable is true
- `chapter` must reference an existing Chapter

## Landing Page Element
**Description**: Components of the AI-animated landing page that showcase humanoid robotics capabilities

**Fields**:
- `id` (string): Unique identifier for the landing page element
- `animationType` (enum): One of ["3D-model", "2D-animation", "interactive", "static-visual"]
- `title` (string): Title of the element
- `description` (text): What the element demonstrates about humanoid robotics
- `technology` (string): Implementation technology (e.g., "Three.js", "Unity WebGL")
- `interactionType` (enum): One of ["passive", "interactive", "guided"]
- `animationDuration` (integer): Duration in seconds for the animation
- `responsive` (boolean): Whether the element adapts to different screen sizes

**Validation Rules**:
- `animationType` must be one of the allowed values
- `interactionType` must be one of the allowed values
- `animationDuration` must be positive

## User Progress
**Description**: Tracks learner progress through the book modules and chapters

**Fields**:
- `userId` (string): Unique identifier for the user
- `chapterId` (string): Reference to the Chapter being tracked
- `status` (enum): One of ["not-started", "in-progress", "completed"]
- `timeSpent` (integer): Time spent in seconds
- `lastAccessed` (datetime): Timestamp of last interaction
- `completionDate` (datetime): Timestamp when status became "completed"
- `notes` (text): Optional user notes about the chapter
- `rating` (integer): Optional rating from 1-5

**Validation Rules**:
- `status` must be one of the allowed values
- `rating` must be between 1 and 5 if provided
- `chapterId` must reference an existing Chapter

## Search Index
**Description**: Index for fast content search across the entire book

**Fields**:
- `id` (string): Unique identifier for the search entry
- `title` (string): Title of the indexed content
- `content` (text): Processed text content for search
- `url` (string): Relative URL to the content page
- `module` (string): Module ID where content is located
- `chapter` (string): Chapter ID where content is located
- `tags` (array of strings): Associated tags for filtering
- `relevanceScore` (float): Precomputed relevance score for search results

**Validation Rules**:
- `url` must be a valid relative URL
- `module` must reference an existing Book Module
- `chapter` must reference an existing Chapter

## Glossary Term
**Description**: Definitions of key terms used throughout the book

**Fields**:
- `id` (string): Unique identifier for the term
- `term` (string): The term being defined
- `definition` (text): Clear, concise definition
- `module` (string): Primary module where term is introduced
- `relatedTerms` (array of strings): Other glossary terms related to this one
- `examples` (array of strings): Usage examples
- `category` (string): Category like "ROS", "Simulation", "AI", etc.

**Validation Rules**:
- `term` must be unique across all glossary terms
- `definition` must not exceed 500 characters