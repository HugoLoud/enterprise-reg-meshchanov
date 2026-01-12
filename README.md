# Enterprise RAG Challenge — Solution

This repository contains an individual solution for the Enterprise RAG Challenge.

## Description

The system implements a Retrieval-Augmented Generation (RAG) pipeline for answering questions
based on annual report PDFs.

Pipeline:
1. PDF parsing and text chunking
2. Embedding with Sentence Transformers
3. FAISS vector index
4. (Optional) Cross-Encoder reranking
5. Answer generation (heuristic / LLM)
6. Automatic submission file generation

## Setup

Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Create a .env file with required variables:
GENERATOR=openai
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.mistral.ai/v1
OPENAI_MODEL=mistral-small-latest
TEAM_EMAIL=your_email
SUBMISSION_NAME=surname_v1

Run:
python main.py
This will generate a submission file in the submissions/ directory.

The system supports heuristic, OpenAI-compatible, and Gemini generators.
