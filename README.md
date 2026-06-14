# Threat Detection AI

A starter project for a threat detection and risk assessment system inspired by *Person of Interest*.

## Core Architecture

- Data ingestion: email, chat, calls, social media, incident reports
- Speech-to-text: phone call transcription with speaker ID, timestamps, language detection
- Entity extraction: people, locations, dates, organizations, phone numbers, addresses, vehicles, weapons
- Knowledge graph: relationships between entities and events
- Risk modeling: threat, harassment, fraud, extremist rhetoric, violent intent
- Alert engine: explainable risk levels with source traceability
- Timeline reconstruction: event sequencing for context
- Natural language query layer for analysts

## Project Structure

- `src/threat_ai/`
  - `ingest.py` — ingest and normalize multi-source data
  - `stt.py` — speech processing and transcript normalization
  - `entities.py` — entity detection and relationship extraction
  - `graph.py` — build and query a knowledge graph
  - `risk.py` — generate risk scores and alerts
  - `timeline.py` — reconstruct event timelines
  - `query.py` — natural language query support
  - `pipeline.py` — orchestrates the full processing flow
  - `models.py` — domain models for documents, entities, alerts, and events

## Getting Started

1. Create a Python environment: `python -m venv .venv`
2. Activate it: `source .venv/Scripts/activate` (Windows PowerShell) or `.\.venv\Scripts\Activate.ps1`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the API: `uvicorn app:app --reload`

## Reproducible Evidence Demo

Run a deterministic synthetic challenge set and produce scorecards, red-team
robustness metrics, an investigation report, and visual-proof data:

```bash
python deployment_runner.py --seed 12345
```

The runner writes artifacts to `evidence_output/`:

- `challenge_documents.jsonl` - detector inputs with labels withheld
- `ground_truth_labels.json` - hidden scenario labels for evaluation
- `scorecard.txt` and `scorecard.json` - precision, recall, lead time, and scenario recall
- `investigation_report.json` - explanation for a flagged hidden scenario
- `visual_proof.json` - network graph, timeline, and risk trajectory data

Synthetic evidence proves the architecture works inside the simulated world. It
does not prove real-world predictive performance without public, consented, or
research datasets with known outcomes.

## Notes

This project is a starting point for building a scalable threat intelligence system with explainable alerts and cross-source correlation.
