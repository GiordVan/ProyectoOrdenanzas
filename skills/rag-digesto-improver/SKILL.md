---
name: rag-digesto-improver
description: Improve and debug the Villa Maria digital digest RAG assistant (retrieval, prompts, ambiguity handling, and citations). Use when asked to improve answer quality, enforce clarification questions when context is insufficient, prioritize year in tariff and budget queries, or tune backend/frontend chat behavior.
---

# RAG Digesto Improver

## Quick Start

1. Read `backend/chat_engine.py` and `backend/api.py`.
2. If chat UX is involved, read `frontend/src/components/ChatIA.tsx` and `frontend/src/components/BuscadorIA.tsx`.
3. Read [domain-rules.md](references/domain-rules.md).
4. Apply minimal, testable changes. Avoid speculative refactors.

## Non-Negotiable Behavior

1. Answer using retrieved context only.
2. If context is insufficient, ask one short clarification question instead of guessing.
3. If the question is about amounts, rates, taxes, fees, or budget and no year is provided, ask for year first.
4. Keep citations/documents aligned with what was really used.
5. Keep outputs concise and citizen-friendly.

## Backend Workflow

1. Add deterministic guardrails before model calls.
2. Keep `/ask` and `/ask-stream` behavior consistent.
3. Preserve current response schema:
`{"respuesta": "...", "ordenanzas_citadas": [...]}` for `/ask`.
4. Keep latency stable: avoid expensive per-request scans if cache or preloaded metadata can be reused.

## Frontend Workflow

1. Ensure streaming clients consume `chunk` events correctly.
2. Surface clarification questions as normal assistant output.
3. Do not block document preview when clarification is returned.

## Validation Checklist

1. Syntax-check edited Python files.
2. Run golden tests from `rag-digesto-tester` skill when available.
3. Manually test:
- Pricing question without year -> clarification question.
- Pricing question with year -> direct answer.
- Budget 2025 total -> explicit amount.

