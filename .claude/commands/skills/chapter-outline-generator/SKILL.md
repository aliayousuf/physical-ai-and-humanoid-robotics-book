
name: chapter-outline-generator
category: pedagogical
applies_to: part-level-planning
required_for:
  - chapter-planner
  - content-implementer
description: >
  Generate clear, pedagogically sound chapter outlines from part-level intent.
  This skill defines why each chapter exists, how it fits into the narrative,
  and how it should be taught, ensuring controlled cognitive load and
  readiness for lesson-level implementation.

skill:
  title: Chapter Outline Generator
  purpose: >
    Transform part-level intent into structured, learner-centered chapter
    outlines that provide unambiguous guidance for authors, educators,
    and downstream lesson planners.

core_objectives:
  - Translate part intent into chapter-level structure
  - Make the chapter’s purpose explicit and defensible
  - Control cognitive load and concept density
  - Define clear learning outcomes and success criteria
  - Ensure outlines are immediately implementable

guiding_principles:
  - Specification-First:
      intent: Chapters are designed before any prose is written
      rule: No chapter drafting without an approved outline
  - Narrative Continuity:
      intent: Chapters advance a learning journey, not a topic list
      rule: Each chapter must justify its position
  - Cognitive Load Discipline:
      intent: Learners should not be overwhelmed
      rule: Limit new concepts per chapter
  - Progressive Scaffolding:
      intent: Support decreases as competence grows
      rule: Early chapters guide heavily; later chapters do not
  - Show-Before-Explain:
      intent: Concrete examples precede theory
      rule: Demonstrate outcomes before abstraction
  - Implementation Readiness:
      intent: Outlines must be actionable
      rule: Another author can implement without clarification

chapter_outline_structure:
  required_sections:
    - chapter_metadata
    - chapter_purpose
    - narrative_role
    - learning_outcomes
    - section_flow
    - cognitive_load_profile
    - practice_design
    - success_criteria
    - chapter_connections

chapter_metadata:
  required_fields:
    - chapter_number
    - chapter_title
    - complexity_tier: [Beginner, Intermediate, Advanced, Professional]
    - estimated_duration

learning_outcomes:
  constraints:
    - Maximum of 5 outcomes
    - Observable and testable
    - Action-oriented verbs
    - Directly tied to chapter purpose

section_flow_patterns:
  default:
    - Orientation (why this chapter matters)
    - Demonstration (show the result)
    - Explanation (core concepts)
    - Guided practice
    - Independent verification
    - Reflection
  alternatives:
    - Problem → Attempt → Resolution
    - Example → Principle → Application
    - Case Study → Analysis → Generalization

cognitive_load_profile:
  fields:
    - load_level: [Light, Moderate, Heavy]
    - new_concept_count
    - scaffolding_level: [High, Medium, Low]
  rules:
    - Beginner chapters: Light or Moderate load only
    - Heavy load requires explicit justification

practice_design:
  requirements:
    - At least one hands-on activity
    - Clear learner responsibility
    - Clear system or tool responsibility
    - Explicit validation step
    - Reflection prompt included

chapter_connections:
  required_links:
    - previous_chapter: Assumed knowledge
    - next_chapter: Enabled capability
    - internal_references: Concepts reused later

workflow:
  step_1:
    name: Interpret Part Intent
    actions:
      - Identify learning goal of the part
      - Determine target learner level
      - Extract prerequisite knowledge
  step_2:
    name: Define Chapter Purpose
    actions:
      - Answer “Why does this chapter exist?”
      - Assign complexity tier
      - Define success conditions
  step_3:
    name: Design Section Flow
    actions:
      - Select appropriate flow pattern
      - Order sections to minimize cognitive load
      - Place examples before abstractions
  step_4:
    name: Specify Outcomes and Practice
    actions:
      - Write measurable learning outcomes
      - Design hands-on practice
      - Define validation and reflection
  step_5:
    name: Validate Outline Quality
    actions:
      - Check concept density
      - Verify narrative continuity
      - Confirm implementability

quality_standards:
  must:
    - Explicit chapter purpose
    - Clear narrative role
    - Controlled concept density
    - Practice with validation
    - Measurable success criteria
  acceptance_checks:
    - Outline can be implemented without follow-up questions
    - Chapter fits logically between neighbors
    - No hidden assumptions
    - Ready for lesson-level breakdown

success_metrics:
  - metric: Implementability
    description: Authors can write the chapter directly
  - metric: Narrative Clarity
    description: Learners understand why the chapter exists
  - metric: Cognitive Load Control
    description: Pacing feels manageable
  - metric: Planning Efficiency
    description: No rework required downstream

anti_patterns:
  - Chapter as topic dump
  - Implicit or vague goals
  - Excessive new concepts
  - Missing practice or validation
  - Outline that functions only as a table of contents
