# vuln-cost

Minor Thesis — Vulnerability Detection Cost Comparison

Measuring the token cost of LLM and LLM-agent approaches to software
vulnerability detection, across different code-context strategies.

## Setup

Requires Python 3.12.14 (installed via `brew install python@3.12`).

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

PrimeVul (Ding et al., ICSE 2025) — https://github.com/DLVulDet/PrimeVul

Not tracked in this repository. Download the release artifact and place the
`.jsonl` files in `data/`.

Splits used:
- `primevul_valid_paired.jsonl` - development and trial run
- `primevul_test_paired.jsonl` - final reported results

## Layout

- `data/` - PrimeVul dataset (gitignored)
- `logs/` - raw per-call logs, JSONL (gitignored)
- `results/` - derived tables and metrics (tracked)