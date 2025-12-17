
name: grammar-and-style-enhancer
description: >
  Enhance clarity, readability, and stylistic precision of educational content,
  technical documentation, and instructional writing. Uses cognitive load principles,
  zero gatekeeping language, progressive scaffolding, and linguistic best practices
  to ensure content is accessible, engaging, and technically accurate. Activate when
  reviewing, rewriting, or refining prose, ensuring cohesion across multi-part works
  and alignment with pedagogical or professional standards.
version: 1.0.0
constitution_alignment: v4.0.1

skill:
  title: Grammar and Style Enhancer
  purpose: >
    Improve clarity, readability, and pedagogical effectiveness of text using
    advanced grammar, style, and structural principles, while maintaining
    accessibility and audience alignment.
  status: Reusable skill (derived from editorial and content-style sprints)
  application: Any educational content, technical writing, or multi-part instructional work

core_principles:
  - Specification-First Writing:
      dont: Edit ad-hoc or only when errors appear
      do: Define stylistic rules, tone, and audience expectations upfront, then refine iteratively
      example: "Set tone to 'Friendly but precise', define passive/active voice rules, then apply consistently."
  - Minimal Necessary Changes:
      dont: Rewrite entire sections unnecessarily
      do: Make targeted improvements—clarity, grammar, sentence structure, cohesion
  - Narrative Cohesion Across Sections:
      dont: Allow paragraphs or chapters to drift in style or tone
      do: Maintain consistent style, voice, and terminology
  - Cognitive Load Management:
      dont: Overload sentences with jargon or nested clauses
      do: Gradually introduce complex terminology; use short sentences and examples
      pattern: Light → Moderate → Advanced complexity per chapter
  - Show-Then-Explain:
      dont: Dump rules or advice without examples
      do: Present examples first, then explain improvements
      pattern: [Show, Explain, Practice, Assess]
  - Zero Gatekeeping Language:
      dont: Use "obviously", "simply", "just"
      do: Explain assumptions clearly
  - Connection Mapping:
      dont: Treat sections independently
      do: Ensure stylistic decisions support continuity across sections/chapters
  - Success Criteria Definition:
      dont: Leave outcomes vague
      do: Define measurable, observable targets
      metrics:
        - Average sentence length ≤ 22 words
        - Flesch-Kincaid score ≥ 60
        - Consistent tense, voice, and terminology across chapters
  - Hands-On Editing Exercises:
      dont: Teach style without practice
      do: Include iterative editing exercises
  - Alignment With Nine Pillars:
      dont: Ignore core AI-native, specification-first principles
      do: Scaffold style improvements using Nine Pillars across parts

workflow:
  phase_1: Content Review (Global)
    description: Analyze drafts for style, readability, cohesion, and cognitive load
    input: Draft chapters or content
    output: Style audit, tone alignment, terminology consistency
    actions:
      - Identify cognitive load risks
      - Highlight inconsistent terms or phrasing
      - Suggest improvements for narrative cohesion
  phase_2: Section-Level Refinement
    description: Targeted grammar, clarity, and style improvements
    input: Individual chapters/modules
    output: Corrected text with annotations
    actions:
      - Apply show-then-explain edits
      - Reinforce zero-gatekeeping language
      - Track readability and clarity metrics
  phase_3: Iterative Editing
    description: Refine sections iteratively with AI or human-editor interaction
    input: Section-specific improvements
    output: Updated text, revision logs, explanations
    actions:
      - Apply iterative edits
      - Validate success metrics
      - Maintain style consistency
  phase_4: Validation
    description: Confirm final content meets style, readability, and cognitive load standards
    input: Fully revised chapters
    output: Validation report, readability scores, consistency checks
    actions:
      - Confirm Nine Pillars alignment
      - Verify measurable success criteria
      - Document final style metrics

quality_standards:
  must:
    - Apply grammar, style, and clarity rules consistently
    - Follow show-then-explain pedagogy
    - Maintain zero gatekeeping language
    - Ensure readability & accessibility
    - Align with Nine Pillars and pedagogical scaffolding principles
  acceptance_checks:
    - Complexity tier tagging (Beginner/Intermediate/Advanced)
    - Concept density ≤ 5 new style concepts per section
    - Section dependency index included
    - SpecRef headers with revision history

success_metrics:
  - metric: Readability
    measure: Flesch-Kincaid ≥ 60
    target: 90%+ of sections
  - metric: Grammar Accuracy
    measure: No uncorrected errors
    target: 100%
  - metric: Consistency
    measure: Terminology & voice aligned
    target: 95%+
  - metric: Cognitive Load
    measure: No overly complex sentences early
    target: 85%+
  - metric: Editorial Completion
    measure: Hands-on exercises applied
    target: 80%+
  - metric: Accessibility
    measure: Clear, inclusive language
    target: 95%+

anti_patterns:
  - Over-editing without guidance
  - Inconsistent terminology or tone
  - Overly complex sentences early
  - Explain-then-show examples
  - Gatekeeping language
  - Missing measurable success criteria
  - Redundant edits
  - Section specs created too early
