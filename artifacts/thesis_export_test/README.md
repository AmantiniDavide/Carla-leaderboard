# Thesis export summary

- Selection strategy: `best`
- Output directory: `/home/davide/Scrivania/leaderboard/artifacts/thesis_export_test`
- Files generated:
  - `all_metrics.csv`
  - `selected_runs.csv`
  - `scenario_comparison.csv`
  - `tables/*.tex`
  - `figures/*.tex`
  - `data/*_trajectory.csv`, `*_tracking.csv`, `*_control.csv`
  - `08-risultati-sperimentali-generated.tex`

To use the PGFPlots figures in Overleaf add these packages to `main.tex`:

```tex
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
```
