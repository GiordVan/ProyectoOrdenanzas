---
name: rag-digesto-tester
description: Run regression tests for the Villa Maria digest RAG assistant using a golden question set. Use when asked to test answer quality, verify budget/tariff responses, check clarification-question behavior, or validate backend changes in /ask and /ask-stream.
---

# RAG Digesto Tester

## Quick Start

1. Ensure backend API is running.
2. Run:
`python skills/rag-digesto-tester/scripts/run_golden_eval.py --base-url http://localhost:8000`
3. Review pass/fail output and fix failures in backend retrieval or prompting logic.

## Test Objective

1. Validate direct-answer cases for budget and governance.
2. Validate clarification-question behavior for ambiguous pricing/tariff queries.
3. Catch regressions after prompt/retrieval/frontend changes.

## Golden Set

Question set file:
`skills/rag-digesto-tester/assets/golden_questions.json`

Supported expectation types:

1. `clarification`: response should be a clarification question and include required terms.
2. `contains_all`: response should include all required patterns/phrases.

## Exit Criteria

1. All golden tests pass.
2. Any new domain rule is reflected in `golden_questions.json`.
3. Failures include reproducible question + response in report.

