<!--
Sync Impact Report:
- Version change: undefined → 1.0.0
- Added sections: All principles and sections based on Physical AI & Humanoid Robotics educational framework
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ no changes needed (generic template)
  - .specify/templates/spec-template.md ✅ no changes needed (generic template)
  - .specify/templates/tasks-template.md ✅ no changes needed (generic template)
- Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics Book Constitution

## Core Principles

### Decision Point Mapping
Every chapter and learning activity must identify critical decisions learners must make, distinguishing between decisions requiring human reasoning versus those suitable for agent execution. Content must provide decision frameworks with criteria, constraints, examples, and context-specific prompts.

### Reasoning Activation
Content must force learners to think and reason rather than mimic or passively consume. Instruction adapts across comprehension, application, analysis, and meta-cognition layers. Learners must continuously build meta-awareness of what they're doing, why, and how to improve their reasoning.

### Intelligence Accumulation
Every chapter must accumulate reusable intelligence by reusing context from earlier chapters and producing outputs that become future inputs (skills, patterns, tools, sub-agents). Chapters function as progressive intelligence scaffolds, not isolated knowledge units.

### Right Altitude
Avoid extremes: too low (rigid, prescriptive instructions) or too high (vague directives). Content must maintain 'just right' decision frameworks with concrete reasoning prompts, examples, constraints, and adaptive pathways.

### Frameworks Over Rules
Avoid hard rules; prefer conditional reasoning that adapts to the learner's clarity, goals, and state of reasoning. Use conditional structures like 'If the learner lacks clarity on what is being built (spec), delay showing implementation (code).'

### Meta-Awareness Against Convergence
Actively disrupt convergence toward predictable teaching patterns such as lecture formats, taxonomy-driven sequencing, toy examples, and passive explanation. Maintain adaptive variability through Socratic dialogue, discovery tasks, spec-first building, error analysis, collaborative debugging, and multi-agent reasoning activities.

## Technical Requirements

The published textbook must include an embedded RAG chatbot, built with OpenAI Agents / ChatKit SDKs, FastAPI backend, Neon Serverless Postgres (for structured knowledge), and Qdrant Cloud Free Tier (for vector retrieval). Capabilities required: answer questions about the book's content, operate as a reader-assistant that responds based only on selected text when required, and serve as a living, spec-driven companion demonstrating agentic software principles.

## Development Workflow

All content follows Spec-Driven Development (SDD) methodology. Every chapter must include decision frameworks, reasoning activation, and intelligence accumulation. Content must be built using Spec-Kit Plus tools and follow agentic development paradigms.

## Governance

This constitution supersedes all other practices for the Physical AI & Humanoid Robotics book project. All content must comply with the core principles of Decision Point Mapping, Reasoning Activation, Intelligence Accumulation, Right Altitude, Frameworks Over Rules, and Meta-Awareness Against Convergence. Amendments require documentation of impact on learning outcomes and pedagogical effectiveness.

**Version**: 1.0.0 | **Ratified**: 2025-12-08 | **Last Amended**: 2025-12-08