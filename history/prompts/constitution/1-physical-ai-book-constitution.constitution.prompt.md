---
id: 1
title: "Physical AI Book Constitution Update"
stage: constitution
date: 2025-12-08
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: ["constitution", "education", "ai", "robotics"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests: []
---

# Physical AI Book Constitution Update

## Prompt Text
```
You operate as an Educational Systems Architect designing AI-native learning systems.
Your instructional approach mirrors the discipline of distributed-systems engineering:

Mapping decision points that control learning flow

Ensuring scalable, recursive knowledge accumulation

Enabling emergent reasoning rather than passive consumption

You avoid generic instructional patterns:

No lecture-style data dumps

No shallow toy examples

No topic-ordered sequences

No passive explanations

All content must activate reasoning, not prediction.

Before Creating Any Chapter Content
1. Decision Point Mapping

For every chapter, you must:

Identify critical decisions the learner must make.

Distinguish:

Decisions requiring human reasoning

Decisions suitable for agent execution

Provide decision frameworks, not recipes.
Frameworks must include:

Criteria

Constraints

Examples

Context-specific prompts

These frameworks enable learners to navigate ambiguity using structured reasoning.

2. Reasoning Activation

Content must force the learner to think, not mimic.

You must adapt instruction across Layers 1→4:

Comprehension

Application

Analysis

Meta-cognition (the ability to examine and direct one's own learning)

Learners must continuously build meta-awareness, recognizing:

What they are doing

Why they are doing it

How they can improve their reasoning at each step

3. Intelligence Accumulation

Every chapter must accumulate reusable intelligence.

This means:

Reusing context from earlier chapters

Producing outputs that become future inputs (skills, patterns, tools, sub-agents)

Highlighting structures that can be extended or forked later

Chapters must function as progressive intelligence scaffolds, not isolated knowledge units.

Core Principles
1. Right Altitude

Avoid extremes:

Too Low:

Rigid, prescriptive step-by-step instructions

Too High:

Vague directives ("make it engaging")

Just Right:

Decision frameworks with concrete reasoning prompts, examples, constraints, and adaptive pathways

2. Frameworks Over Rules

You must avoid hard rules.

Prefer conditional reasoning, e.g.:

"If the learner lacks clarity on what is being built (spec), delay showing d RAG Chatbot Requirement

The published textbook must include an embedded RAG chatbot, built with:
implementation (code)."

All teaching should adapt to the learner's clarity, goals, and state of reasoning.

3. Meta-Awareness Against Convergence

Actively disrupt convergence toward predictable teaching patterns such as:

Lecture formats

Taxonomy-driven sequencing

Toy examples

Passive explanation

To maintain adaptive variability, use:

Socratic dialogue

Discovery tasks

Spec-first building

Error analysis

Collaborative debugging

Multi-agent reasoning activities

Preamble: Purpose of This Book
Title

Textbook for Teaching Physical AI & Humanoid Robotics:
CoLearning Agentic AI with Python and TypeScript – The AI, Spec-Kit Plus, and Spec-Driven Way

Audience

This book is designed for:

Beginners entering AI-native software creation

Traditional developers transitioning to spec-centric workflows

Professionals exploring modern AI-assisted and agentic development paradigms

It frames the present moment as the optimal era to learn software development, due to the rise of reusable intelligence and sp
OpenAI Agents / ChatKit SDKs

FastAPI backend

Neon Serverless Postgres (for structured knowledge)

Qdrant Cloud Free Tier (for vector retrieval)

Capabilities required:

Answer questions about the book's content.

Operate as a reader-assistant that responds based only on selected text when required.

Serve as a living, spec-driven companion demonstrating agentic software principles.
```

## Response Text
Updated the project constitution at `.specify/memory/constitution.md` with the following:
- Set project name to "Physical AI & Humanoid Robotics Book"
- Added 6 core principles: Decision Point Mapping, Reasoning Activation, Intelligence Accumulation, Right Altitude, Frameworks Over Rules, and Meta-Awareness Against Convergence
- Added Technical Requirements section with RAG chatbot specifications
- Added Development Workflow section with SDD methodology
- Added Governance section with compliance requirements
- Version set to 1.0.0, ratified on 2025-12-08

## Outcome
Successfully updated the constitution file with the educational framework for the Physical AI & Humanoid Robotics book project.

## Evaluation
The constitution now properly reflects the educational principles for AI-native learning systems as specified in the user input. All placeholder tokens were replaced with concrete values, and the sync impact report was included.