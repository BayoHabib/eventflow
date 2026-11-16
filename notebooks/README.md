# Notebooks

## `demo_chicago_crime.ipynb` — UNVERIFIED, not executed in CI

This notebook has never been executed in CI or in any audit. Its figures —
spatial correlation ρ = 0.515, graph–sequence correlation r = 0.953,
104 anomalies (84 spikes, 20 drops) in Q1 2024, reverse-causality claim —
are **neither confirmed nor refuted** (see
`../docs/audit/known-limitations.md`). Do not quote them as facts.

### Last execution attempt (2026-11-16)

Environment: a scratch venv with the repo installed editable plus
`nbconvert`, `ipykernel`, `matplotlib`, `networkx`, `seaborn`
(the shared dev venv has no Jupyter stack). Command:

```bash
jupyter nbconvert --to notebook --execute demo_chicago_crime.ipynb \
  --output /tmp/nbexec-output.ipynb --ExecutePreprocessor.timeout=300
```

Result: **exit code 1**. Cells 0–2 (imports) run; the data-load cell
fails with:

```
ValueError: cannot concat empty list
  at crime_df = pl.concat(dfs)
```

Cause: the notebook reads monthly CSVs from
`../../chicago_crime_data_cli/data/monthly_2020_2025/monthly/{2024-01,2024-02,2024-03}_chunk_0001.csv`
— a directory from the author's local machine that does not exist in
this environment, so no file is found and `dfs` is empty. The Q1 2024
data exists on disk here only as a single file,
`../../data/raw/chicago_crime/crimes_q1_2024.csv` (the `make audit`
replay input), in a layout this notebook does not read.

### What would make it executable

Point the data-load cell at a directory with the monthly
`{month}_chunk_0001.csv` layout, or rewrite it to read the replay CSV —
and then re-verify every figure before quoting it. The notebook's stored
outputs are from the author's Windows run (2025-12-08) and were not
reproduced here. The recipe itself, by contrast, is replayed on real
Q1 2024 data by `make audit`.
