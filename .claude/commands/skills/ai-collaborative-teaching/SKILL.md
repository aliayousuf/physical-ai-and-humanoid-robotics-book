

name: ai-collaborative-teaching
category: pedagogical
applies_to: all-chapters
required_for:
  - chapter-planner
  - content-implementer
description: >
  Design AI-native learning experiences where humans and AI collaborate as
  bidirectional learning partners. This skill formalizes co-learning pedagogy,
  specification-first collaboration, and ethical AI use patterns to prepare
  learners for real-world AI-driven development workflows. It ensures students
  build independent capability while effectively leveraging AI as teacher,
  learner, and co-worker.
version: 2.1.0
dependencies:
  - constitution:v4.0.1

skill:
  title: AI Collaborative Teaching
  purpose: >
    Enable educators to design structured co-learning environments where AI
    actively participates as a learning partner rather than a passive tool.
    Emphasizes specification-first thinking, validation-before-trust, and
    progressive learner autonomy.

core_objectives:
  - Teach specification-first collaboration as the primary AI skill
  - Establish bidirectional learning between human and AI
  - Balance AI assistance with independent skill development
  - Promote ethical, transparent, and verifiable AI use
  - Prepare learners for professional AI-native development workflows

three_roles_framework:
  ai_roles:
    - Teacher:
        description: Suggests patterns, architectures, and practices unfamiliar to learners
    - Student:
        description: Learns from human domain expertise, feedback, and corrections
    - Co-Worker:
        description: Collaborates as a peer, contributing ideas but not owning decisions
  human_roles:
    - Teacher:
        description: Provides specifications, constraints, and domain context
    - Student:
        description: Learns from AI suggestions and alternative approaches
    - Orchestrator:
        description: Designs collaboration strategy and makes final decisions
  requirement: >
    Every AI-integrated lesson must explicitly demonstrate at least two AI roles
    and two human roles in action.

convergence_loop:
  description: Required iterative learning pattern for all AI-assisted lessons
  steps:
    - Human specifies intent with context and constraints
    - AI proposes approach or solution
    - Human evaluates and learns from AI output
    - AI adapts based on human feedback
    - Human and AI converge on improved outcome
  non_negotiables:
    - Iteration is required
    - Perfection on first attempt is disallowed
    - Both human and AI must add unique value

pedagogical_requirements:
  per_lesson:
    - At least one insight learned from AI
    - At least one AI adaptation based on learner feedback
    - Explicit validation step before accepting output
    - Reflection on what improved through collaboration
  forbidden_patterns:
    - AI as passive autocomplete
    - One-way instruction (human → AI or AI → human only)
    - Unverified AI outputs
    - Hidden learning from AI suggestions

graduated_teaching_alignment:
  tiers:
    tier_1:
      focus: Foundational skills
      ai_usage: None or minimal
      goal: Independent competence
    tier_2:
      focus: Complex execution
      ai_usage: Assisted
      goal: Spec → Generate → Validate mastery
    tier_3:
      focus: Scale and orchestration
      ai_usage: High
      goal: Strategic oversight and supervision
  note: >
    This skill defines HOW AI is used once AI involvement is appropriate,
    not WHEN to introduce AI.

ai_assistance_balance_models:
  standard: { foundation: 40, ai_assisted: 40, verification: 20 }
  beginner_heavy: { foundation: 60, ai_assisted: 20, verification: 20 }
  advanced_heavy: { foundation: 25, ai_assisted: 55, verification: 20 }

workflow:
  step_1:
    name: Context Analysis
    actions:
      - Identify learner level and topic
      - Define independent skills that must be protected
      - Identify ethical and integrity constraints
  step_2:
    name: Specification Design
    actions:
      - Define learning outcomes
      - Assign AI and human roles per activity
      - Define validation criteria
  step_3:
    name: Collaboration Pattern Selection
    actions:
      - Select AI role (explainer, debugger, reviewer, pair)
      - Define convergence loop checkpoints
  step_4:
    name: Lesson Construction
    actions:
      - Foundation phase (no AI)
      - AI-assisted exploration phase
      - Independent verification phase
  step_5:
    name: Validation and Iteration
    actions:
      - Assess balance ratios
      - Verify independent capability
      - Adjust AI involvement as needed

ethical_guidelines:
  principles:
    - Transparency of AI use
    - Attribution and disclosure
    - Understanding over output
    - Bias awareness
    - Professional accountability
  enforcement:
    - Require explanation of AI-generated work
    - Include AI-free assessments
    - Document AI usage decisions

quality_standards:
  must:
    - Explicit spec → generate → validate loop
    - Demonstrated bidirectional learning
    - AI-free verification checkpoints
    - Ethical guidelines enforced
    - Learner agency preserved
  acceptance_checks:
    - AI roles explicitly identified
    - Balance model documented
    - Validation prompts included
    - Reflection activity present

success_metrics:
  - metric: Independent Competence
    target: 80%+ learners succeed without AI
  - metric: Learning Convergence
    target: 85%+ reach defined outcomes
  - metric: Ethical Compliance
    target: 100%
  - metric: Engagement
    target: 90%+ active participation
  - metric: AI Adaptation Quality
    target: 85%+

anti_patterns:
  - AI-first before foundations
  - No independent verification
  - Treating AI as authority
  - Hidden or undisclosed AI use
  - Over-reliance on generated code
  - Vague specifications
